# MP32 MegaMoE 初始设计与布局证据

- [最初设计草案](MP32_MEGA_MOE_PREFILL_DESIGN.md)：实现前确认的 FP8×FP4、40 MPs、full pool、三阶段与 gate32/up32 方案。
- [Gate/up 布局分析](INTERLEAVE_ANALYSIS.md)：RS `partition_C(identity)` 的线程和寄存器配对推导。
- [当前实现与验证](../../mp32_mega_moe_prefill.md)：实际接口、内核文件、mutlass pin、编译资源、测试及全 token 最终 y 精度结果。

前两份文档保留初始设计语境。初稿中的文件命名、prefill-only 分配和跨 warp amax scratch 是候选设计；当前实现复用 MP31 IPC/workspace 分配，并通过 64 KiB BF16 activation staging 完成 K32 量化。精确接口和已验证范围以当前实现文档为准。

## 重现布局枚举

使用仓库固定的 mutlass `d2ef2bb1257bba5731927e88b46e2118fa5d281a`，在仓库根目录执行：

```bash
/usr/local/musa/bin/mcc -x musa -std=c++17 -O2 --offload-arch=mp_32 \
  -I3rdparty/mutlass/include \
  docs/design/mp32_mega_moe/probe_interleave.cpp \
  -o /tmp/mp32_gate_up_probe
/tmp/mp32_gate_up_probe > /tmp/mp32_mma_coordinates.txt
python docs/design/mp32_mega_moe/analyze_interleave.py \
  /tmp/mp32_mma_coordinates.txt
```

该程序只枚举 host 端 MMA 坐标；检查完整 256×256 tile 覆盖、32,768 个 RS gate/up pairs 同线程且 fragment 差值为 32，以及 K32 group 的四 warp ownership。检查成功时输出 `PASS: tile coverage, all pairing permutations, and K32 quant ownership`。生成的坐标和日志不提交到仓库。
