# 小 batch MLA 与 JOYAI TP2 复现入口

这里是测试用接入层，不修改 SGLang 安装目录。主核和 combine 的实现位于 MATE 正常源码目录。优化针对 MP31、固定 query、B<=5、Q*Hq<=32；JOYAI TP2 普通 decode 每卡 H=16、Q=1，适用该路径。其余调用回退到原 SGLang 函数，prefill 保持默认。

运行时代码：

- [csrc/mla_pybind.mu](../../csrc/mla_pybind.mu)：M32/M128 分发。
- [主循环](../../include/mate/attention/flash_mla/fmha_mla_collective_tme_warpspecialized.hpp)：QK/softmax 与 PV 分组、P 双缓冲及在线缩放。
- [线程块调度](../../include/mate/attention/flash_mla/mla_kernel_tme_warpspecialized.hpp)：加载组、计算组、pipeline 与 epilogue。
- [并行 combine](../../include/mate/attention/flash_mla/mpxx_mla_combine_parallel.hpp) 和 [combine 分发](../../csrc/attention_combine.mu)。

| 文件 | 用途 |
|---|---|
| `adapter.py` | 唯一的固定 query MLA FFI 封装、shape/dtype 判断及 decode hook |
| `run_server.py` | default / optimized / validate 三种入口，参数通过环境传给 TP 子进程 |
| `validate_real_attention.py` | 同一实际 Q/KV 的 A/A、A/B、FP32 reference；始终传播默认输出 |
| `profiling.py` | pinned SGLang 版本的 MUSA profiler 采集兼容 |
| `build_library.py` | 从当前优化源码编译库并打印路径与 SHA256 |
| `bench_kernel.py` | 显式加载两库，在相同 metadata 下进行算子 graph A/B |
| `prepare_requests.py` | 从旧脚本生成一次请求并固定为 fixture |
| `run_benchmark.py` | 导入旧压测模块，替换请求生成器为 fixture，保留 HTTP/并发/计时口径 |
| `measure_server.py` | 预热、清缓存、3轮正式测试；结束后才跑生成探针和 profiler |
| `analyze_results.py` | 校验输入一致性、汇总 default_final/optimized_final 与 trace |
| `request_precision.py` / `check_numerics.py` | 触发并检查真实模型 attention 数值验证 |

## 环境

已验证：S5000 MP31、MUSA4.3.8、torch2.9.0.post1、tvm-ffi0.1.11.post1+musa.1；SGLang `63bd31efce00cd302f6ca8ac584fa8ae179bc8d0`。服务默认使用已安装 MATE0.2.7；优化库来自本仓 `bbcd8b8c715eabddf663a30e000e0f2ebcb39bc3` 加本次补丁。需要初始化本仓 MuTLASS 子模块。

以下命令在匹配环境/容器内执行。路径均由调用者设置；服务进程不要把优化 MATE 仓加入 PYTHONPATH，否则默认组也会切换到修改后的 Python 包。

```bash
export MATE_REPO=/path/to/optimized/mate
export JOYAI_PYTHON=/path/to/matching/python
export JOYAI_MODEL_PATH=/path/to/JoyAI-LLM-Flash
export JOYAI_SGLANG_SOURCE=/path/to/pinned/sglang
export JOYAI_GPUS=4,5
export JOYAI_PORT=38642
export RESULT_ROOT=/path/to/new/results
export HISTORICAL_DIR=/path/to/JD_Joy
export TOOL_DIR="$MATE_REPO/benchmarks/joyai_mla"
mkdir -p "$RESULT_ROOT"
```

`HISTORICAL_DIR` 应包含原始 `benchmark_serving_parallel.py` 和 `backend_request_func_parallel.py`，不需要复制或修改它们。旧脚本的 `request_rate` 是 `Parallel(n_jobs=...)` 固定并发，不是 QPS。旧脚本按流式文本事件计数，输出事件可能少于请求的300 token；空响应会被整理后的入口拒绝。

## 编译与算子测试

```bash
cd "$MATE_REPO"
MATE_MUSA_ARCH_LIST=3.1 MATE_WORKSPACE_BASE="$RESULT_ROOT/build" \
  PYTHONPATH="$MATE_REPO" "$JOYAI_PYTHON" "$TOOL_DIR/build_library.py" \
  --record "$RESULT_ROOT/library.json"
# 将上面打印的library路径设为：
export MATE_MLA_LIBRARY=/absolute/path/to/mla_ops.so

MUSA_VISIBLE_DEVICES=4 MATE_MUSA_ARCH_LIST=3.1 \
  MATE_WORKSPACE_BASE="$RESULT_ROOT/build" PYTHONPATH="$MATE_REPO" \
  "$JOYAI_PYTHON" -m pytest -q tests/test_mla_combine.py \
  tests/test_mla_pipeline_overlap.py tests/test_joyai_mla_adapter.py
```

算子 A/B 使用独立原版 checkout 编译得到的原库，不把旧库覆盖到当前 JIT cache：

```bash
"$JOYAI_PYTHON" "$TOOL_DIR/bench_kernel.py" \
  --baseline-library /path/to/original/mla_ops.so \
  --optimized-library "$MATE_MLA_LIBRARY" --output "$RESULT_ROOT/kernel" \
  --heads 16 --batches 1 2 3 4 5 --lengths 6000 7000 8000 9000 10000
```

## 固定输入的服务端 A/B

只生成一次请求，两个版本复用同一文件。要复现已测数据，可直接使用原实验目录保存的 `canonical_requests.json`；重新生成的 fixture 不保证与旧 fixture 字节一致。

```bash
PYTHONPATH="$JOYAI_SGLANG_SOURCE/python" "$JOYAI_PYTHON" \
  "$TOOL_DIR/prepare_requests.py" --historical-dir "$HISTORICAL_DIR" \
  --tokenizer "$JOYAI_MODEL_PATH" --output "$RESULT_ROOT/requests.json"
```

在一个终端启动服务，在另一个终端测量；默认组测完后终止本次服务，在同一对空闲卡启动优化组。不要在抓过 profiler 的同一进程中再收集正式性能数据。

```bash
bash "$TOOL_DIR/launch.sh" default
# 另一个终端：
PYTHONPATH="$JOYAI_SGLANG_SOURCE/python" "$JOYAI_PYTHON" \
  "$TOOL_DIR/measure_server.py" --output "$RESULT_ROOT/default_final" \
  --requests "$RESULT_ROOT/requests.json" --historical-dir "$HISTORICAL_DIR" \
  --tokenizer "$JOYAI_MODEL_PATH"

# 终止上述默认服务后：
bash "$TOOL_DIR/launch.sh" optimized --mla-library "$MATE_MLA_LIBRARY"
# 另一个终端：
PYTHONPATH="$JOYAI_SGLANG_SOURCE/python" "$JOYAI_PYTHON" \
  "$TOOL_DIR/measure_server.py" --output "$RESULT_ROOT/optimized_final" \
  --requests "$RESULT_ROOT/requests.json" --historical-dir "$HISTORICAL_DIR" \
  --tokenizer "$JOYAI_MODEL_PATH"

"$JOYAI_PYTHON" "$TOOL_DIR/analyze_results.py" \
  --results "$RESULT_ROOT" --requests "$RESULT_ROOT/requests.json"
```

每次测量使用新输出目录，防止与旧结果混合。默认地址为127.0.0.1:38642；改变服务端口时给测量脚本传相同的 `--url`。

## 真实 Q/KV 数值验证

先终止性能测试服务，再启动 validate；该模式关闭 graph，不计时。请求的4个输出 token 会因框架预执行而产生4次实际 decode 执行。

```bash
bash "$TOOL_DIR/launch.sh" validate --mla-library "$MATE_MLA_LIBRARY" \
  --validation-output "$RESULT_ROOT/numerics" --disable-cuda-graph
# 另一个终端：
"$JOYAI_PYTHON" "$TOOL_DIR/request_precision.py" \
  --requests "$RESULT_ROOT/requests.json" --output "$RESULT_ROOT/numerics-response.json"
"$JOYAI_PYTHON" "$TOOL_DIR/check_numerics.py" \
  --input "$RESULT_ROOT/numerics" --output "$RESULT_ROOT/numerics-summary.json"
```

该检查对相同实际输入重复默认 kernel、调用优化 kernel、计算 FP32 reference；默认输出传给下一层和下一步，避免生成分叉改变待比较输入。首层额外核对 CPU FP32。使用原 MATE reference 容差 atol=0.015、rtol=0.01，并报告实际误差。通过只代表这些 attention 输入的数值检查，不代表模型任务准确率通过。贪心文本和分叉后的 logprob 不能代替数值验证。

已测结果：旧口径固定输入的 TP2 平均 TPOT 11.225→9.858 ms，下降12.18%；原真实输入320组数值检查通过，A/A全部逐位相同，A/B最大relative L2为0.02494%。这些是原测量记录，整理代码不会重新定义性能数字。详细原始记录保存在工作区的 `joyai-mla-tp2-20261019/`，新复测写入调用者指定的结果目录。
