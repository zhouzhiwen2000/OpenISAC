# Changelog

本文件记录 OpenISAC 的重要更新。

当前从以下提交开始整理：

- Commit: `fe0ee68b67d6621ed22c3eaa71d1de0a7490ce3c`
- Date: `2026-04-02 22:59:43 +08:00`
- Subject: `Improve overflow/underflow recovery, add macOS support, benchmark scripts, and configurable data resource blocks`

## 2026-04-04 - 资源映射与紧凑感知更新

### Summary

本次更新围绕通信/感知资源映射、紧凑感知输出，以及配套文档可读性展开，重点是让资源配置更明确、compact sensing 更易用。

### Changes

- `data_resource_blocks` 新增 `kind` 概念，支持 `payload` 与 `sensing_pilot` 两类资源块。
  影响：可以在同一帧内显式区分“真正承载通信数据的 RE”和“保留给感知参考的已知 RE”，更方便做通信/感知资源共存实验。

- 新增 `sensing_mask_blocks`，并配合 `sensing_output_mode=compact_mask` 用矩形块定义单站/双站感知真正导出的 RE。
  影响：感知输出不再只能走 dense 全缓冲模式，可以只发送感兴趣的 RE，降低输出带宽并提高配置灵活性。

- 当 `sensing_mask_blocks` 满足规则采样条件时，运行时 `MTI` 和本地 Delay-Doppler 处理可以开启。
  影响：规则 compact 配置既能保留紧凑原始 RE 输出，也能在本地继续生成 Delay-Doppler 结果。

- 新增 `config/*_ResourceMap.yaml` 示例配置，展示 `data_resource_blocks`、`sensing_pilot` 与 `sensing_mask_blocks` 的典型写法。
  影响：不再需要完全从零手写复杂 YAML，复现实验配置更直接。

- Web 配置工具新增 `Sensing Resource Map` 视图，并增强 `Resource Planner`，让 `data_resource_blocks` 与 `sensing_mask_blocks` 都可以在时频平面中可视化编辑。
  影响：资源规划从“手写坐标”进一步变成“可视化拖拽”，减少配置错误。

- 新增 `scripts/sensing_runtime_protocol.py`，并让 fast viewer 使用运行时参数握手来识别 compact sensing 输出元数据。
  影响：后端与前端之间对 compact sensing 的元数据约定更清晰，后续扩展 viewer 更容易。

- README 与文档页补回 changelog 链接，并澄清 `data_resource_blocks` / `sensing_mask_blocks` 的职责区别，以及当前 viewer 对非规则 compact payload 的限制。
  影响：新用户更容易理解两个资源映射参数分别控制什么，以及当前前端能处理到什么程度。

### Notes

- `data_resource_blocks` 在 TX 与 RX 侧应保持一致，包括每个块的 `kind`。
- `sensing_mask_blocks` 仅在 `sensing_output_mode=compact_mask` 时生效。
- 当前 `plot_sensing*.py` 与 `plot_bi_sensing*.py` 只能处理“规则”的 compact payload；非规则选择仍需自行解析。

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
