# JIT 转储与回放

pytest 用例准备调用某个 JIT 内核时，MATE 可以先保存 `kernel.so`、输入数据和调用
参数，而不真正执行这次调用。生成的产物可以脱离 Torch 和 MATE 单独回放；回放时只需
安装兼容版本的 TVM-FFI Python 包。

目前只有 FMHA 前向路径接入了这套工具。

## 通过 pytest 生成转储

运行 pytest 时，通过 `--jit-dump` 指定产物的保存目录。一次可以选择一个或多个测试
项：

```bash
pytest -s \
  --jit-dump /tmp/fmha-artifacts \
  'tests/test_fmha.py::test_varlen_func_bshd[param-1]' \
  'tests/test_fmha.py::test_varlen_func_bshd[param-2]'
```

这个选项实际设置的是 `MATE_JIT_DUMP_DIR`。直接设置环境变量也能得到相同的结果：

```bash
MATE_JIT_DUMP_DIR=/tmp/fmha-artifacts \
pytest -s 'tests/test_fmha.py::test_varlen_func_bshd[param-1]'
```

如果要为其他架构生成产物，可以通过 `MATE_MUSA_ARCH_LIST` 指定目标架构：

```bash
MATE_MUSA_ARCH_LIST=3.2 pytest -s \
  --jit-dump /tmp/fmha-artifacts \
  'tests/test_fmha.py::test_varlen_func_bshd[param-1]'
```

生成 JIT dump 时，如果设置了 `MATE_MUSA_ARCH_LIST`，其中必须恰好包含一个目标；未
设置时默认使用当前设备架构。产物中会记录三个架构字段：

- `dump_arch`：由 `MATE_MUSA_ARCH_LIST` 选择；未设置该变量时使用当前设备架构。
- `config_arch`：算子选择配置和 dispatch 分支时所依据的架构。
- `binary_arch`：`kernel.so` 的实际编译架构。

如果 `dump_arch` 不是当前设备架构，算子接入代码必须分别给出 `config_arch` 和
`binary_arch`。以 FMHA 为例，`metadata` 会按 `dump_arch` 选择配置，但它的二进制
仍针对当前设备编译；前向内核则按 `dump_arch` 选择配置并编译二进制。

环境变量必须在 pytest 开始收集测试前设置，因为 FMHA 模块会在导入时决定是否启用
转储。

每当测试项准备调用已接入的 JIT 内核时，工具就会生成一个编号目录。编号从 `0000`
开始，按执行顺序递增。如果某个测试项没有产生转储，pytest 会发出警告，但测试结果
本身不会改变。

```text
/tmp/fmha-artifacts/
├── manifest.json
├── 0000/
│   ├── DUMP_COMPLETE
│   ├── kernel.so
│   ├── loader.so
│   ├── runtime/
│   ├── tensor_*.bin
│   ├── replay.py
│   ├── benchmark.py
│   ├── compare.py
│   └── run.sh
└── 0001/
    └── ...
```

`manifest.json` 记录全局的 `dump_arch`，以及每个编号目录对应的 pytest node ID、
FFI 函数、`config_arch` 和 `binary_arch`。

保存目录已经存在时，工具只会发出警告，不会拒绝运行。当前用例的生成文件会被覆盖，
目录中的其他文件不会被修改。

这套工具不能和 pytest-xdist 一起使用，但普通 pytest 进程可以一次选择多个测试项。

## 保存参考结果

如果回放后还需要检查数值结果，可以使用 `--jit-dump-reference` 保存参考结果
（reference）：

```bash
pytest -s \
  --jit-dump /tmp/fmha-artifacts \
  --jit-dump-reference \
  'tests/test_fmha.py::test_varlen_func_bshd[param-1]'
```

也可以直接设置 `MATE_JIT_DUMP_REFERENCE=1`。测试代码需要在调用目标函数之前计算并
注册 reference：

```python
from mate.jit.dump import register_dump_reference

out_ref, lse_ref = compute_reference_before_target_call()
register_dump_reference("out", out_ref, atol=1.5e-2, rtol=1e-2)
register_dump_reference("lse", lse_ref, atol=1.5e-2, rtol=1e-2)
call_target_kernel()
```

如果还要和另一份实现的误差做对比，可以同时注册基线结果（baseline）：

```python
from mate.jit.dump import register_dump_reference_with_baseline

register_dump_reference_with_baseline(
    "out",
    out_ref,
    baseline,
    max_error_factor=4,
    mean_error_factor=3,
)
```

注册 `reference` 时使用的名称，必须和调用目标 FFI 函数时声明的输出名称一致。
`reference` 和 `baseline` 的 shape 也必须和对应输出相同。一次转储只记录当前 FFI
调用的直接输出，因此不能用后续 combine 得到的结果进行比较。

## 逐个回放用例

回放需要安装 TVM-FFI Python 包。以下版本已经过验证：

```bash
python3 -m pip install \
  'apache_tvm_ffi==0.1.9.post3+musa.1' \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

回放时应使用与转储环境相同的 TVM-FFI 版本，否则 Python 扩展可能与 `loader.so` 的
ABI 不兼容。进入一个编号目录后，可以分别执行回放和比较：

```bash
cd /path/to/fmha-artifacts/0000
./run.sh replay
./run.sh compare
```

不带参数执行 `./run.sh` 时，会依次完成这两步。`replay` 加载 `kernel.so`，按保存的
参数调用目标函数，同步后将结果写入 `output_*.bin`。`compare` 只读取已有输出和
`references.json`，不会再次运行内核。

`replay` 返回 0 表示回放完成；`compare` 返回 0 表示所有已注册的比较均已通过。

每个编号目录只对应一个用例。请逐个运行和排查，不要批量回放多个目录。

## 对单个用例做性能测试

artifact 可以在不安装 Torch 和 MATE 的环境中测试所转储 TVM-FFI 调用的性能。warmup
调用全部完成后才开始计时；每次 profile 调用使用一对 MUSA event 计时，所有 profile
调用提交后只执行一次设备同步：

```bash
cd /path/to/fmha-artifacts/0000
./run.sh bench --warmup-iters 10 --profile-iters 100
```

`--warmup-iters` 默认是 10，可以设为 0；`--profile-iters` 默认是 100，并且必须大于
0。命令输出一个 JSON 对象，其中包含以微秒为单位的最小值、中位数、平均值、p90 和
最大设备执行时间。

传入单次调用的理论浮点运算量后，还会报告逐次测量的吞吐率分布：

```bash
./run.sh bench --warmup-iters 10 --profile-iters 100 --flops 26388279066624
```

此时 JSON 还会包含 `workload_flops` 和 `throughput_tflops`。TFLOPS 使用十进制定义
（`FLOP/s / 10^12`）；调用方需要根据被 dump 算子的计算口径提供正确的 FLOP 数。

benchmark 模式只加载一次输入，并在每次调用中复用同一组 Tensor。它不会保存输出，
也不会执行 reference 比较。因此，被转储的函数必须能够使用相同参数安全地重复调用。
当前 benchmark 不会 flush L2 cache。

## Loader 与运行时

生成产物时，MATE JIT 会编译 `tools/jit_dump/loader.cpp`，并将当前使用的 MUSA
运行时库一并打包。loader 提供 Tensor 读写、设备同步和 MUSA event 计时。自动编译
需要完整的 MATE 源码；如果环境中只有安装好的 MATE Python 包，可以通过以下环境变量
提供预编译文件：

```bash
MATE_JIT_DUMP_LOADER_SO=/path/to/loader.so
MATE_JIT_DUMP_MUSART_SO=/path/to/libmusart.so.4
```

也可以手动编译 loader：

```bash
tools/jit_dump/build_loader.sh /tmp/loader.so
```

目前，所有 Tensor 参数都必须位于同一块 MUSA 设备上，并且内存连续。生成产物时不会
实际调用目标 FFI 函数。
