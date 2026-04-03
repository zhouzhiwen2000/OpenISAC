# Changelog

本文件记录 OpenISAC 的重要更新。

当前从以下提交开始整理：

- Commit: `fe0ee68b67d6621ed22c3eaa71d1de0a7490ce3c`
- Date: `2026-04-02 22:59:43 +08:00`
- Subject: `Improve overflow/underflow recovery, add macOS support, benchmark scripts, and configurable data resource blocks`

## 2026-04-02 - fe0ee68

### Summary

本次更新主要提升了系统稳定性、跨平台可用性、实验可测性，以及资源配置和前端适配能力。

### Changes

- 改进了异常后的恢复链路，重点增强了 underflow / 时序异常后的 TX burst 重启、frame slot 跳过、frame sequence 维护，以及 RX 侧基于时间戳的重新配对和 frame boundary 自动校正。
  影响：单站感知在异常恢复后更不容易因为收发失步出现时延偏差、峰值漂移或无峰值现象，降低了非实时系统调度抖动对结果的影响。

- 感知链路新增基于 `frame_seq` 的 RX/TX 成对匹配机制，并在序号不一致时主动丢弃过期帧而不是错误配对。
  影响：恢复场景下的感知结果更稳健，减少了“拿错 TX 参考帧”带来的错误时延与错误峰值。

- 增加 macOS，尤其是 Apple Silicon 的本地构建支持，补充了 CMake 依赖发现逻辑和独立构建文档。
  影响：项目不再基本局限于 Ubuntu，mac 上也可以进行本地开发、编译验证和演示。

- 感知显示前端增加 Apple GPU 加速路径，`plot_sensing_fast.py` 和 `plot_bi_sensing_fast.py` 支持通过 `MLX` 使用 macOS 的 Metal 后端，并保留自动 CPU fallback。
  影响：macOS 上的快速显示不只是“能运行”，而是可以获得明显更好的实时性和交互体验。

- 新增可配置的 `data_resource_blocks`，支持用矩形时频资源块精确定义 payload 占用区域。
  影响：可以更方便地做保护带、稀疏资源映射、部分子载波/符号承载数据等实验，也更容易控制通信与感知资源分配。

- Web 配置工具新增 `Resource Planner`，支持直接在时频网格上绘制 payload 资源块，并提供 `Guard Band Grid` 预设。
  影响：资源规划从“手写 YAML”变成了“可视化编辑”，明显降低配置复杂度和出错率。

- 新增内部测量模式，支持用确定性 PRBS 载荷进行在线 BER / BLER / EVM 统计。
  影响：通信链路质量评估更标准化、更自动化，便于做参数 sweep 和可重复实验。

- 新增 benchmark 脚本，包括 BER / BLER / EVM sweep、解调端端到端时延测试、以及调制端感知运行时 benchmark。
  影响：性能评估和回归验证更系统，后续优化可以更快量化收益。

- profiling 更细化，调制端、解调端、感知处理链都可以输出更明确的分阶段耗时统计。
  影响：更容易定位瓶颈是在数据摄取、IFFT/FFT、LDPC、还是 sensing pipeline 本身。

- 新增每通道系统时延估计模式，以及按通道启停 sensing output 的能力。
  影响：更适合做单独的系统时延标定和分通道实验，不必强制走完整感知输出链。

### Notes

- `data_resource_blocks` 启用后，TX 与 RX 的配置应保持一致；若与 `sync_pos` 或 `pilot_positions` 重叠，后两者优先。
- macOS 当前主要面向本地开发和演示；后端实时调优脚本仍以 Linux 为主。
- 前端 GPU 后端优先级现在为：CUDA > Apple MLX > Intel GPU > CPU fallback。
