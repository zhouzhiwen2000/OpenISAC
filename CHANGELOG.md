# Changelog

本文件记录 OpenISAC 的重要更新。

当前从以下公开版本开始整理：

- Public baseline: [fe0ee68b67d6621ed22c3eaa71d1de0a7490ce3c](https://github.com/zhouzhiwen2000/OpenISAC/commit/fe0ee68b67d6621ed22c3eaa71d1de0a7490ce3c)
- Date: `2026-04-02 22:59:43 +08:00`
- Subject: `Improve overflow/underflow recovery, add macOS support, benchmark scripts, and configurable data resource blocks`

## 2026-10-23 - UDP 链路测试与实时展示

新增 C++ `UdpLinkTest`，发送可配置速率的已知 UDP 数据包，统计接收吞吐、丢包率、误比特率及重复、乱序等情况。支持 ZeroMQ 实时统计与在线码率控制，新增 BS 端 PyQt 性能面板，展示收发流量曲线和用户可输入的 LDPC 码率所对应的编码后速率，默认码率为 `1/2`。补充中英文使用说明。

## 2026-10-23 - 大频偏捕获后的连续跟踪

修复 UE 在捕获并补偿大频偏后，正常跟踪仍可能选择错误频偏分支并重复调频的问题。跟踪保留初始捕获与后续有效观测的频偏参考，按实际接收调谐变化更新残余频偏；重捕时清理旧参考，忽略调频前的排队观测。补充外部变频器场景下关闭 CFO 时延预测的配置说明。

## 2026-10-23 - 单站速度显示与微多普勒历史

单站快速感知界面新增可选 `Velocity RF Carrier (GHz)` 输入，填写实际射频载频后，距离谱、微多普勒图和检测目标列表统一显示径向速度（m/s）；留空恢复多普勒频率（Hz）。按单站往返关系换算并保留多普勒符号，设置独立于 USRP 中频和调谐参数，自动保存供下次启动使用。主谱图右上角单独以黄色文字显示最强检测目标的距离及速度或频率，保留一位小数。单站快速上位机保留丢帧和帧抽取前后的微多普勒滚动历史，按时间戳对缺失采样补零，避免反复清空图像；采样几何变化或时间戳回退时重建历史。

## 2026-10-23 - 在线感知帧抽取

新增 `sensing.frame_decimation` 和上位机 `Frame every N` 在线控制，每 N 个完整感知帧处理并输出 1 帧，默认 1。保留帧内 STRD、连续信道估计与 MTI，跳过后续距离–多普勒处理和数据发送，并在观测空档处重新积累微多普勒历史。同步配置编辑器与中英文参数说明；感知参数协议更新为 V10，后端与上位机需配套更新。

## 2026-10-20 - 独立 XR1710G 接收 UE 指南

将 XR1710G 接入说明整理为中英文独立页面，按设备准备、数据源配置、BS/UE 参数匹配、启动停止和可选硬件频偏校正组织操作步骤。快速开始末尾提供入口并注明测试阶段适用范围，更新导航、README 和参数参考链接。

## 2026-10-15 - 双站感知 LoS 相位参考

新增启动配置 `sensing.bistatic_phase_reference`，改善独立或稳定性较差时钟下残余公共相位漂移造成的感知相干性下降和多普勒不稳定。启用要求收发机之间存在稳定且占主导的 LoS 路径，收发机相对静止；感知目标可以运动。此选项适用于 `dense` / `los_tracking` 感知，保留幅度，输出相对于参考路径的多普勒，不改变通信 IQ 和频偏控制。无效参考、慢时间采样缺口或显示坐标改变时重建感知窗口及历史。

CPU、CUDA 和 ROCm 使用一致的配置和行为。新增中英文参数说明、Web 配置字段及使用条件说明。

## 2026-10-15 - XR1710G UE 接收后端

新增 `radio_backend: xr1710g`，UE 可读取 Linux 主机上 `w1700k-stream` 的单路 IQ 共享内存，支持 40、80、160 和 320 MS/s。后端提供 IQ 转换、软件频偏校正和局部采样时间戳，在丢包、覆盖或数据源重启后沿用 UE 的失步重捕流程，支持关闭上行运行。

新增可选的 `xr1710g_pll_command`，通过 PLL 微调实现硬件载频校正，保留原有过时频偏观测过滤、缓冲样本和时间戳。PLL 调节不校正采样时钟。支持 AVX2/FMA 的主机可使用 SIMD 加速 IQ 转换和软件混频。

提供 `UE_XR1710G.yaml`、接收探针和转换性能工具，并补充 Web 配置与中英文接入说明。此后端仅接收；射频配置由外部控制程序管理，不支持上行、PPS 或双路交织流。

## 2026-10-13 - 明确视频推流与播放步骤

重写中英文视频工作流，按 BS 电脑推流、USRP 空口传输、UE 电脑播放的顺序说明数据路径、命令执行位置和 UDP 端口。首次 OTA 的视频步骤统一使用带发送节奏控制的 MPEG-TS/UDP 命令，明确下方命令中视频文件路径的替换方式，补充对应播放命令、异机播放设置，并提供详细参数说明的入口。

## 2026-10-13 - 补充 UDP 视频突发限速说明

中英文视频工作流补充 FFmpeg MPEG-TS-over-UDP 推流与匹配播放命令，说明 `bitrate`、`burst_bits`、包长和缓冲的单位与选值方法，区分编码目标、UDP 发送节奏及实测净荷吞吐。首次 OTA 视频步骤增加对应入口，补充关键帧突发、封装余量与接收验证说明。

## 2026-10-13 - 恢复 BS 重启后的双站感知流

双站感知检测输入帧符号编号回退，重置采样游标并丢弃旧时间线的未完成批次，清除 MTI 与微多普勒历史，迁移待生效参数；保留当前 STRD、MTI 和 FFT 配置。修复 BS 重启后通信已恢复、双站感知却等待旧采样编号而暂停的问题，CPU 与 GPU 路径行为一致。正常帧仅增加时间戳比较与记录。

新增真实 ZeroMQ 输出回归，覆盖时间线重启、旧批次丢弃、待生效参数迁移、跨帧大 STRD 和 MTI 历史清理；更新中英文故障排查说明。

## 2026-10-13 - 校验 UE 前导并恢复重连后的感知控制

UE 在提交同步帧边界前，检查配置要求的重复 ZC 与 CFO 训练字段；全局相关峰落在前一个重复 ZC 时，继续验证后一个符号，拒绝不完整或不匹配的前导。运行中的无效前导在解调和感知前丢弃，连续失败达到现有 `reset_hold_s` 后重新搜索同步，保留原有频率、队列世代、OCXO hold 与冷却处理。ZC-only 配置保留原同步行为。

快速感知界面监测 ZeroMQ 控制连接断开与重连，清除旧参数和显示缓存，重新握手后下发一次保存的 MTI、STRD 等控制；普通心跳不重复下发。新增带噪声、频偏、CP 时延偏移和不完整前导检查，以及真实 ZeroMQ 后端重启回归。更新中英文故障排查与参数参考。

## 2026-10-12 - 修正 Hsys 配置读取并补充时延校准步骤

Hsys 命令行工具读取当前 YAML 的 `ofdm_frame.fft_size`，保留旧 `ofdm` 节兼容性；感知显示的补零 FFT 不参与 Hsys 文件几何校验。回归检查覆盖两端配置、节名优先级和真实控制传输。中英文校准流程补充重复同步符号导致相关峰歧义时的处理，以及恢复运行增益后的适用性检查。

## 2026-10-12 - 优化 CPU LDPC 默认译码参数

CPU LDPC 默认使用 12 次最大迭代与 0.75 NMS 系数，统一到 YAML 缺省值、编解码器配置与工厂、全部 BS/UE 模板、Web 编辑器和 CPU 精度基准。保持 syndrome 提前结束，已有 YAML 的显式值继续生效。补充缺省配置与显式覆盖回归检查，并更新中英文参数参考、Web 使用说明及生成站点。

## 2026-10-12 - CPU LDPC 参数吞吐基准

`LDPCFixedBench` 支持设置最大迭代次数、NMS 系数、计时 Es/N0 和重复次数，并检查参数范围，保留原有前三个位置参数。吞吐按多次计时中位数报告，同时输出最佳值、BER/FER 与机器可读结果；明确区分信息位吞吐、编码位吞吐和 Q16 量化串行估算。补充中英文复现命令，可比较默认与调节后的 CPU 译码参数。

## 2026-10-12 - YAML 与 Web 配置 CPU LDPC 译码参数

新增 `ldpc.decoder_iterations` 和 `ldpc.nms_normalization_factor`，应用于 UE 下行与 BS 上行的 float32/Q16 CPU 译码。默认保持 6 次最大迭代、NMS 系数 1.0 和 syndrome 提前结束；YAML 与 Web 保存检查迭代次数、有限值范围及 Q16 支持的系数。所有 BS/UE 模板和中英文参数文档同步更新，Web 的 Q16 比例控件随定点开关显示，修改参数后重启后端生效。

Web 停止后端使用 SIGINT 并等待清理，确保 UHD 收发正常结束。新增旧配置兼容性、非法输入保护、Web YAML 往返和带噪声实际译码回归检查。

CPU LDPC 在设置 SIMD 批次数后显式重置全部译码状态，避免新分配的消息缓冲影响首次 float32/Q16 译码；独立精度基准也使用相同初始化。回归检查验证编码码字的 syndrome，并分别覆盖迭代次数、NMS 系数和两者组合的干净载荷译码。

## 2026-10-12 - 修正 Aff3ct 安装编译参数

中英文安装教程将 `-funroll-loops -march=native -faligned-new` 合并到一次 `CMAKE_CXX_FLAGS` 设置，确保本机 SIMD 优化与对齐内存选项完整传入 Aff3ct。补充迁移到不同 CPU 指令集时需要重新编译的说明。

## 2026-10-11 - OpenISAC 论文正式发表

### Summary

更新中英文 README 与网站首页的论文状态和引用：OpenISAC 已正式发表于 *IEEE Internet of Things Journal*，vol. 13, no. 17, pp. 41254–41269, Sept. 1, 2026，doi: [10.1109/JIOT.2026.3710751](https://doi.org/10.1109/JIOT.2026.3710751)。采用正式期刊题名，并增加 IEEE Xplore 入口。

## 2026-09-15 - 命令行 Hsys 校准与系统时延日志分析

### Summary

新增 `calibrate_hsys.py`：显式选择单站／双站逻辑感知通道，使用 V9 握手和实际工作目录 YAML 检查运行参数，备份旧校准文件，并等待新文件完成角色、通道、FFT、射频参数、符号数和有限值校验；支持离线文件质量摘要。不自动改变或恢复增益，超时／中断不误报完成。

新增 `calibrate_system_delay.py`，按通道分析最近的系统时延日志，通过相关峰质量、样本数与建议值极差筛选稳定结果；重启或 alignment 变化后重置统计，不自动修改 YAML。补充中英文使用流程和离线回归测试。Hsys 命令行流程借鉴 Henry Teall <henry.teall@paconsulting.com> 的 [9b630f3](https://github.com/philippaPA/OpenISAC/commit/9b630f3744a2261b234d22f95bfe6e07b727d4d8)，不引入 DC 基线方案。

## 2026-09-15 - 感知物理坐标与 backend viewer 修复

### Summary

修复 backend sensing viewer 缺失的目标区域范围回调，恢复窗口正常初始化。fast/backend viewer 统一使用 Doppler Hz、单站距离 m、双站相对时延 ns 和微多普勒时间 s；坐标换算使用原始 OFDM FFT、CP、实际 sensing stride 与处理 FFT 长度，并保持裁剪、补零和显示降采样下的标定一致。物理坐标相对于当前同步／系统时延对齐参考。

感知参数协议统一为 V9（104 字节），CPU、CUDA、ROCm 发送端同步更新，viewer 拒绝旧版／未知参数包；非均匀子载波集合不伪造物理距离。Doppler 坐标与回调修复参考 philippaPA/OpenISAC 的 `cca9d5e` 和 `9b630f3`，扩展为共用坐标逻辑与单站／双站物理单位。

## 2026-08-05 - 系统时延校准日志绕过日志 gate

### Summary

当感知通道设置 `enable_system_delay_estimation: true` 时，CPU、CUDA 和 ROCm 的 `SYSDLY` 校准消息现在绕过模块日志过滤和实时调用点限流，确保校准建议值持续可见。

## 2026-07-29 - 稳定 UHD 定时发送并保留网卡多队列

### Summary

CPU、CUDA 与 ROCm 的 BS 下行及 UE 上行 UHD 路径改为每个连续 burst 只在首个 `send()` 上设置 `has_time_spec`；正常连续帧依靠样本数推进时间，underflow/restart 后的新 burst 再提交一次绝对时间戳，避免一次启动抖动被后续逐帧时间戳放大为持续 `TIME_ERROR`。`isolate_cpus.py` 的单核 IRQ 隔离模式不再把高速网卡压缩为一个 combined 队列，而是默认保留当前队列数；只有显式传入 `--nic-combined N` 或选择双队列 RX/TX 分流时才修改队列数。旧 `isolate_cpus.bash` 包装器已删除，benchmark 与文档统一直接调用 `isolate_cpus.py`。

## 2026-07-29 - UHD 收发统一为整帧调用

### Summary

CPU、CUDA 与 ROCm 的 BS/UE 下行、上行及感知路径统一以完整业务帧调用一次 UHD `send()`/`recv()`，不再按 streamer 最大 packet samples 分块或在应用层拼接短返回；UHD 继续在内部完成传输包分片。短收发直接丢弃当前帧并进入现有恢复，定时补零先写入预分配缓冲区后单次发送；共享内存仿真后端保留背压重试。

## 2026-07-29 - 将网卡 IRQ 隔离与收发分流并入 CPU 隔离流程

### Summary

`isolate_cpus.py` 在发现任意内核 `isolcpus` 逻辑核时新增第二级菜单，可选择跳过网卡调优或进行 IRQ 隔离；只有至少两个可用的隔离物理核心时才显示双 combined 队列 RX/TX 分流，将 RX 固定到 queue 0、将 BS/UE YAML 中的 USRP TX 线程固定到 queue 1。IRQ 隔离默认保留当前 combined 队列数，仅在显式传入 `--nic-combined N` 或选择双队列分流时修改；IRQ 核优先选择不同物理核心，并将实际 IRQ CPU 及其 SMT 同胞同时从系统 slice 与后端进程 `AllowedCPUs` 中排除。`set_performance.bash` 不再修改 combined 队列、RSS/XPS、IRQ affinity 或 `irqbalance`，仅保留通用主机、ring 和 MTU 调优，避免两套脚本互相覆盖。

## 2026-07-28 - 增加 Fast FLL 半步控制模式

### Summary

OCXO Fast FLL 新增 `ocxo_fll_fast_control_mode: pi | half_step`。默认 `pi` 保持原 velocity-form PI；`half_step` 不使用 PI 历史，每个 Fast 更新周期直接请求当前 frequency error 的一半，仍受 Fast loop slew、全局 actuator slew 和 DAC command 范围限制。现有默认更新周期保持 0.5 秒，CPU、CUDA 与 HIP UE 共用同一控制状态机和配置解析。

## 2026-07-28 - 短时重新同步保持 OCXO 控制阶段

### Summary

CPU、CUDA 与 HIP UE 在运行中因 delay tracking reset 重新搜索同步时，不再立即把 OCXO discipliner 重置到 Fast。控制器会冻结 reset 前的 Coarse/FllFast/FllSlow/Phase 阶段、控制历史和 DAC command；30 秒内重新获得同步后沿用原阶段，Phase 会等待新 timeline 的有效 TO/PPS anchor 后再解冻。只有重捕获间隔超过 30 秒才执行原有 tracking reset，并重新武装 PPS Phase 对齐流程。`ToStale`、TO discontinuity/jump、frequency-source change、FLL unstable 与 Phase SFO exit 等运行时自动降级条件必须连续维持至少 10 秒才会退回 Fast；短时毛刺只暂停当帧控制，条件恢复后会清零降级计时。

## 2026-07-28 - 修正实时调度继承导致的退出阻塞

### Summary

CPU、CUDA 与 HIP 的 BS/UE 主线程保持普通调度，并在启动引擎后才绑定 `main_cpu_core`，避免 UHD、CUDA 和 ZMQ 辅助线程继承主线程的实时策略与专用 CPU；RX、TX、调制、解调和 LDPC 等数据面线程继续在各自入口显式提升实时优先级并绑定配置核心。BS 停止流程会在任何阻塞操作和线程 join 前先通知异步 TX 线程退出，避免 Ctrl+C 清理阶段被高优先级辅助线程饿死而无法结束。

## 2026-07-27 - 阻止无效 delay 峰触发 UE 错误对齐

### Summary

CPU、CUDA 与 HIP UE 的 delay tracking 现在只有在 delay-spectrum 峰均值有效且 `peak_ratio >= 20` 时才向 SFO estimator 提交观测或调度 RX sample alignment；BS underflow、短暂失锁或噪声帧会清空 alignment 二帧确认并继续既有 reset-hold 判定，不再把 CP 搜索窗内的随机峰放大成错误窗口移动。已实际提交的 alignment 即使其回执帧峰值无效，仍通过独立坐标更新保持 SFO/eRTM delay 连续。Quinn fractional-delay 输出同时限制到文档约定的 `[-0.5, 0.5]`，避免低质量峰产生越界小数观测。

## 2026-07-25 - 收紧多设备 PPS 与 UE timed-rate 前置校验

### Summary

CPU、CUDA 与 HIP 的多设备 PPS 对齐现在要求所有独立 USRP 使用共享的 `external` 或 `gpsdo` 时间源，并在武装设备前拒绝 internal/空时间源；单设备 internal 对齐仍保持支持。`clock_source` 仅接受空值、`internal`、`external` 或 `gpsdo`，拼写错误会给出明确配置错误。B210 TX + X310 sensing 模板统一改用 external PPS。CPU/CUDA/HIP UE 的 timed RX/UL-TX 若设备实际采样率相差超过 1 Hz，现在会在启动任何工作线程前 hard-fail，避免上下行帧身份网格使用不同 rate authority。配置编辑器说明、双语文档、HIP 日志文案及 PPS 回归测试同步更新。

## 2026-07-24 - 由 clock_source 统一触发启动 PPS 对齐

### Summary

CPU、CUDA 与 HIP 的 BS/UE 启动对齐规则统一为：真实 UHD 后端只要解析后的 `clock_source` 非空，就在开流前观察 PPS、执行 `set_time_next_pps(0)` 并验证 device time 复位；`time_source: internal` 也进入该路径，`time_source` 仅选择 PPS 输入。BS 多设备路径不再拒绝 internal time source，但仍保留跨设备时间一致性验证。`BS_B210Tx_X310Rx.yaml` 的全局 `time_source` 改为 `internal`，与 sensing RX 通道保持一致。

## 2026-07-24 - 增加 CUDA 激进混合精度配置

### Summary

CUDA BS/UE 新增单一 `cuda.cuda_precision_profile` 配置；默认 `fp32` 完全保留现有数值路径，`mixed_fp16` 对上下行批量 OFDM FFT/IFFT 和 UE GPU LDPC 消息使用 FP16，同时继续以 FP32执行信道估计、Wiener、CFO/SFO、均衡、噪声与 LLR、MTI 和 sensing 处理。运行时若 cuFFT 不支持当前 FP16 变换形状会自动回退 FP32；CUDA LDPC 仅分配所选解码精度的消息工作区，编码器和控制解析器不再分配未使用的解码工作区。

## 2026-07-20 - 在开流前将 USRP 设备时间对齐外部 PPS

### Summary

当解析后的 `time_source` 为 `external` 或 `gpsdo` 时，CPU/CUDA BS 与 UE 在非实时初始化阶段先观察一次 PPS 跳变，再把所有相关 USRP 同时武装为下一 PPS 边沿 `set_time_next_pps(0)`；BS 的空 `clock_time.time_source` 跟随 `clock_source`，TX、独立 sensing RX 与独立 uplink RX 均进入相同严格路径，UE 则在 RX/UL-TX 开流前完成且不依赖 OCXO mode。启动会轮询到各设备 time counter 确实在目标边沿回跳，再验证 `time_last_pps=0` 与跨设备 `time_now` 一致性；PPS 缺失、backend 不支持或验证失败直接报错，不再退回任意软件时刻的 `set_time_now(0)`。`gpsdo` 路径会有意将原有类 UTC device-time epoch 置零。该步骤统一设备时间零点，但不改变 X310 在 200 MHz master clock 下约 5 ns 的 PPS 边沿捕获量化分辨率。

## 2026-07-20 - 增加 PPS anchor 跳变精确诊断

### Summary

eRTM wire format 升级为 `ERTM5`，在原有 frame/PPS anchor 之外携带 BS 原始 `time_last_pps()` ticks；UE 将该值与本机原始 PPS ns、`r_B/r_U`、tick rate 和 sample rate 一起原子发布。OCXO debug 升级为 `OCXODBG3`：原 16 路 float 通道的索引和含义保持不变，新增 typed diagnostic trailer，CSV 追加精确 `float64/int64` 字段，避免绝对 PPS timestamp 经过 `float32` 丢失低位；自动文件名按 `ocxo_debug_YYYYMMDD_01.csv`、`_02`、`_03` 顺序递增。CPU/CUDA 路径和协议回归测试保持一致，viewer 仍可读取旧 `OCXODBG2` 数据包。

## 2026-07-20 - 每次重新进入 Phase 时执行一次 PPS 调相

### Summary

`pps_phase` 的 PPSDO AP gate 改为在每次 `FllSlow -> Phase` 转换时重新武装，并按最新有效 PPS residual 计算一次调相；同一次 Phase 停留期间保持 one-shot，普通更新和 Freeze 不会重复发送。AP pending/settling 期间继续暂停 Phase SF，旧 AP 尚在 Queued/InFlight 时等待硬件回到 Idle。CPU/CUDA 路径、配置注释、双语文档和确定性重入测试保持一致；保留 `ocxo_pps_initial_ap_*` 键名以兼容现有 YAML。

## 2026-07-20 - 保持 OCXO DAC 值跨重新同步与 UE 重启连续

### Summary

UE/CUDAUE 普通重新同步不再把 OCXO 控制器和 DAC 强制归零，而是清理观测、误差历史和阶段状态并保留当前有效 command；执行器把 slew 时间基准延迟到重新入锁后的第一条实际非零修正，避免长 search 间隔放大首步。串口 worker 在 `SF` 收到 `OK` 后通过 temp fsync、原子 rename 和目录 fsync checkpoint raw DAC word；正常退出时先结束 worker，再保存最后一条已确认的稳态 word，不会把未确认的 queued command 当作有效值。后续 UE/CUDAUE 进程启动仅在串口设备与 DAC 标定元数据一致时恢复并重发同一 word，同时用对应 ppm 初始化控制器；损坏、不匹配或越界状态会被保留供排查，且本次运行禁止覆盖。新增 `ocxo_dac_state_file` 配置，默认写入当前工作目录的 `.openisac_ocxo_dac_state.yaml`，文件 I/O 不进入实时处理线程。

## 2026-07-20 - 更新 OCXO Slow FLL 与 PPS AP 默认参数

### Summary

根据本轮 PPSOUT 台架数据更新 UE OCXO 默认调参：Slow FLL 使用 `kp=0.1`、`ki=0.007`、`0.005 ppm/s` 变化率上限和 `0.0005 ppm` Fast-to-Slow 阈值；初始 PPSOUT AP 保持默认开启，并把已确认的命令极性设为 `+1`。C++ 内建默认值、UE 硬件预设、台架模板、配置编辑器与双语文档保持一致。

## 2026-07-20 - 按 eRTM 报告实时刷新 BS PPS anchor

### Summary

CPU BS/CUDABS 不再让 ERTM4 在整个 timeline epoch 内复用启动时捕获的 PPS anchor；每份达到发送条件的新 eRTM 报告在打包前重新读取 `time_last_pps()`，并把本次快照直接写入 payload。刷新位于 LDPC/eRTM 内部负载线程，不进入 TX/调制实时路径；读取失败时跳过本次报告并按配置的报告帧间隔重试。

## 2026-07-20 - 修正 Mode B TO_clock 控制极性（UE 本振）

### Summary

台架确认：OCXO 在 UE 本振时 Mode A（CFO residual）为负反馈、极性正确；Mode B（`TO_clock` 斜率 FLL / legacy TO Phase）为正反馈。对 Mode B 控制误差取反：FLL 使用 `-to_slope`，legacy Phase 使用 `setpoint - to_clock`；原始 `to_slope_ppm` 仍作诊断；`pps_phase` 误差与 Mode A residual 不变。

### Changes

- `OCXODiscipliner`：`mode_b_frequency_error_ppm()` / `mode_b_to_phase_error_samples()`；Coarse/FLL 与 Phase 诊断 `frequency_error_ppm` 使用 `-to_slope`，legacy Phase/debug `to_err` 使用 `setpoint-to_clock`。
- 更新 `test_ocxo_control.py` 中 Mode B、legacy Phase 与 PPS Phase 的正增益 plant 模型，并增加 Mode B FLL / legacy Phase 闭环收敛回归。

## 2026-07-17 - 新增显式 OCXO PPS Phase 锁定模式

### Summary

保留 `auto`/`phase` 原有的 `TO_clock-setpoint` Phase 锁定，并新增显式 `pps_phase` 模式。BS 在初始化后发布 frame-0/PPS 纪元锚点；UE 用 eRTM `TO_UE`、下行 RF 延迟、同帧边界时间戳和 PPS 时间戳计算 `wrap_1s(P_U-P_B)`，以 0 为固定目标。PPSDO `AP` 每个同步纪元只做一次 PPSOUT 粗调，随后 PLL 继续通过 SF 连续跟踪；AP 不改变 USRP 时间轴、air-frame anchor 或 TO 状态。

### Changes

- eRTM wire format 升级为 `ERTM4`，增加 BS frame-0 相对 PPS 的 tick 距离、帧 tick 周期和 tick rate；拒绝缺少 PPS 锚点的 ERTM2/3，避免协议语义静默混用。
- CPU/CUDA UE 仅在 `pps_phase` 下按 `wrap_1s(tau_DL_RF - TO_UE + r_B - r_U)` 读取和发布 PPS 相位误差；`auto`/`phase` 继续使用 `ocxo_phase_setpoint_mode` 与 `ocxo_phase_setpoint_fixed_samples`，且不读取 `time_last_pps()`。
- UE 空 `time_source` 明确解析为 `internal`；`pps_phase` 强制 `hardware_sync=true`、`ertm_to_enable=true`，并要求 `clock_source` 与 `time_source` 都显式为 `external`，否则配置加载失败。
- 一次性 AP 增加 enable、限幅、稳定时间、完成超时、近零 deadzone 和可台架切换的 polarity；串口队列只合并 SF，绝不丢弃 AP，`JAMSYNCING` 会重试同一请求。
- 修复 AP 入队失败、串口无响应和 Settling 缺少 post-AP publication 会永久阻塞 Phase SF，以及 hard sync reset 后 AP 未正确重新武装的问题。AP worker 使用包含全部 `JAMSYNCING` 重试的总墙钟截止时间，截止后不再发送该 AP；Settling 超时采用有界重试，等待新观测后按残差重算 AP；旧 AP 仍 Queued/InFlight 时新同步纪元等待 Idle，不把 Busy 当作终端失败。
- `pps_phase` 关闭初始 AP 时在配置加载阶段告警一次；AP 运行失败时 CPU/CUDA 日志会报告当前 PPS residual，并明确提示限斜率 Phase PI 可能无法快速消除大残差。
- BS PPS 距离改为整数 tick 模运算，避免大绝对帧号在 `double` 平台丢失精度。OCXO 调试协议升级为 `OCXODBG2`，通道 7 表示当前模式的有效 Phase error。

## 2026-07-16 - OCXO Coarse/FLL/Phase 分层控制与 ZeroMQ 诊断

### Summary

原位替换 OCXO 控制器为 `Coarse -> FllFast -> FllSlow -> Phase` 状态机：Mode A 的 Coarse/FLL 使用解模糊 residual CFO；Mode B 从 Alignment 连续化后的 eRTM `TO_clock` 斜率估计钟 SFO，避免 CFO 多普勒和 OFDM delay-track SFO 的运动分量驱动 OCXO；Phase 仅使用 `TO_clock`。控制器 command、EWMA plant 补偿和粗调剩余量只按硬件实际执行的 delta 推进；CPU/CUDA 统一发布带序号和源时间戳的 16 通道 `OCXODBG1` ZeroMQ 数据包。

### Changes

- normal CFO 按相邻符号相位周期选择最接近当前 tune-system CFO 的分支；Mode A OCXO 输入还需连续 3 帧通过 alias 周期 1/4 的变化门限，避免 cold-start 单点直接驱动粗调。
- eRTM TO 原子快照增加 `A_N`、DL identity 和 sync generation；`TO_clock` 用 `delta_TO_UE - delta_A` 消除已知 Alignment 边沿，同时保留真实 SFO 斜率。
- Mode B 的 Coarse/FLL 与 Phase 进退判据均使用 `TO_clock` 对绝对 frame sequence 的滚动线性回归估计钟 SFO；短窗用于 Coarse/FllFast，slow 窗用于 FllSlow/Phase，避免单报告差分抖动和不等报告间隔偏置。
- Phase 改用继承 FLL command 的增量 PI（velocity form），不再把绝对 PI 输出重复当作 DAC delta 累加；新增闭环收敛测试覆盖该低频振荡回归。
- 根据 100 MHz 硬件记录继续将默认 `ocxo_phase_kp` 调整为 `2.0e-4`，并把 Phase 进入 slope 收紧到 `2.0e-4 ppm`、Slow FLL deadzone 降到 `5.0e-5 ppm`；保持 `ki=1.0e-6`、hold 与 slew 不变，减少反向超调和带残余斜率入锁造成的首个 TO 峰值。
- Freeze 作为纯 overlay 暂停控制计时并延后 TO jump/stale 转移；Coarse 异常退出清理目标并支持 overshoot abort；FLL 保留真实窗口样本，Phase 只在 TO 斜率有效且稳定后进入。
- 新增 Coarse/FLL/Phase YAML、web editor schema、中英文参数文档、ZeroMQ 通道表和支持通道选择/滚动曲线/CSV 记录的 `plot_ocxo_debug.py`；控制、阶段、TO jump 和 Alignment 事件会绕过周期降采样立即发布；删除 Capture/Hold、`ocxo_freq_*` 以及旧 VOFA/FireWater UDP 支持。
- 修复 OCXO viewer 将某个信号分组全部隐藏后再启用时图窗高度塌缩的问题；所有可见分组始终按相同 stretch 均分绘图区。

## 2026-07-16 - 统一绝对帧发送门控与 eRTM 状态

### Summary

CPU/CUDA 共用 `AirFrameTxGate` 与 `ErtmAbsoluteFrameState`，统一帧 claim、generation、stale/future/blank 判定、eRTM 报告缓存、clear generation、`state(M)` 选择和 TO 发布；修复 CUDA 感知帧 identity 漂移及 BS 空帧 claim cursor 并发覆盖。补齐 OCXO 配置 schema 依赖，并修正 CFO EWMA 控制阶跃方向与 Mode B 绝对限幅 anti-windup。

## 2026-07-15 - OCXO 控制律符号与旧 PI 对齐

### Summary

观测 `detected_freq_offset - tune_system_cfo` 与旧版一致（真实系统残差，便于长期把 DSP 补偿卸掉）；dual-mode 相对旧 `update_ocxo_pi_with_error_ppm` 多了一个负号，Mode A/B 命令方向反了。已改回与旧 PI 相同极性（正 residual → 正 OCXO step）。

## 2026-07-15 - Alignment latch / TO 发布审查修复

### Summary

修复 queue-full 后 Alignment 语义不完整、TO 用 payload seq 当版本、sensing/process OCXO 数据竞争、`to_is_new` 被 control gate 吞掉、以及 RX 写 process latch 状态等问题。

### Changes

- **Alignment 侧信道完整语义**：RX 发布 `applied_id/samples/time`；process 在首帧 post-align 注入 SFO 与 eRTM force-remeasure，并清 latch（不依赖队列投递）。
- **TO 发布代**：独立单调 `_ertm_latest_to_ue_pub_gen`，不再用 ERTM3 payload `seq` 判定 Mode B 新报告；`samples`/`pub_gen`/`available` 在 `_ertm_to_publish_mutex` 下一致快照读写。
- **OCXO reanchor**：sensing 仅请求；process 统一 `reanchor_to` / age 清零。
- **`to_is_new`**：`OCXODiscipliner::update` 在 `allow_control=false` 时仍 ingest TO unwrap/slope。
- **线程边界**：RX stream-restart 只 bump `_process_tracking_reset_req`；latch/SFO/OCXO age 由 process 应用。

## 2026-07-15 - OCXO 驯服审查修复

### Summary

修复 dual-mode OCXO 的六项审查问题：TO 陈旧/斜率、Mode A 观测窗、DAC 首次写入、SFO 有效位、发布 docs、计划归档。

### Changes

- **Mode B**：按 eRTM TO *publication generation* 累计 `to_report_age_frames` 并置 `to_is_new`（后修为独立 pub_gen，不再用 payload `seq`）；超龄回退频率环；TO 斜率仅在新报告上用真实报告间隔计算。
- **Mode A**：观测窗必须满 `ocxo_freq_obs_window_s` 才出批，不再被 `obs_count < 2` 短路。
- **执行器**：构造不再发默认 SF；`configure_plant()` 用标定 zero word 首次写入；未配置 plant 时忽略 apply。
- **SFO**：`SFOEstimator::has_estimate()`，首窗回归前不参与融合。
- 重新发布 `docs/`；计划移入 `plans/old/`。

## 2026-07-15 - CFO 预滤波：Adaptive AKF → EWMA

### Summary

将 `AdaptiveCFOAKF` 替换为轻量 `CfoErrorEwma`，仅作 OCXO 观测预滤波；删除自适应 Q/R 与相关 YAML。

### Changes

- **`CfoErrorEwma`**：`filtered ← (1-α)·filtered + α·raw`，并对 OCXO 执行步进做 plant 补偿。
- **配置**：`cfo_ewma_enable` / `cfo_ewma_alpha`（默认 `0.02`）；旧 `akf_*` 键告警忽略，`akf_enable` 可映射到 `cfo_ewma_enable`。

## 2026-07-15 - OCXO 双模式驯服（MV197）

### Summary

移除过快的 legacy OCXO PI 路径，改为匹配精密 OCXO（MV197）慢漂移特性的双模式驯服：无 eRTM 时用 CFO+SFO 做频率对齐；eRTM TO 可用时用解卷绕 `TO_UE` 做时间/相位锁定。执行器使用标定 DAC 映射与硬斜率限制。

### Changes

- **`OCXODiscipliner`**（`OFDMCore.hpp`）：Mode A 频率环 / Mode B 相位环，Capture/Hold/Freeze 状态机；无 legacy 兼容。
- **`HardwareSyncController`**：纯执行器（标定 `ppm_per_lsb` / `zero_word`、硬 slew、串口命令合并）。
- **UE / CUDAUE**：SFO 更新后融合观测并驱动 OCXO；eRTM clear/invalidate 时 re-anchor TO。
- **YAML / web editor / docs**：`ocxo_pi_*` 删除；新增 `ocxo_mode`、`ocxo_dac_*`、`ocxo_freq_*`、`ocxo_phase_*`。旧 `ocxo_pi_*` 键仅告警忽略。

## 2026-07-15 - eRTM 旧 UL 报文绝对空口帧补偿

### Summary

eRTM 不再把“处理时刻”的 DUTI/TADV 与异步旧 UL 信道粗配对。BS 在 **UL Identity MATCH** 后发布绑定测量帧 `M` 与实际 DUTI 的 ERTM3；UE 用 `state_at(M)` 查历史 TADV/Alignment，把旧 UL 谱变换到当前 DL 帧 `N` 的 Alignment 坐标，并用 `DUTI_M/TADV_M` 计算 `tau_c`。无法精确配对时保留上一份有效 TO。

### Changes

- **协议 ERTM3**：在 ERTM2 基础上增加 `timeline_epoch` 与 `ul_measure_frame_seq`；明确拒绝无绝对帧身份的 ERTM2。
- **BS**：UL-RX 将 `applied_duti`、channel、identity 作为同帧 sidecar；仅 Identity MATCH 后通过 `SeqlockedErtmUlMeasurement` 发布；注入时不再读取当前 `_bs_dl_ul_timing_diff`。
- **UE UL-TX**：按绝对帧发布 `UlTxTimingStateEntry` 变更历史（总 shift 改变时下一帧生效；TADV/Alignment 反向相消时当前帧生效）；query/publish/clear 均受同一 mutex 保护，稳定状态使用原子快照 fast path，clear 只做 O(1) 元数据失效，避免非原子 POD 数据竞争和逐帧/长时间锁竞争。
- **UE eRTM**：`RxFrame` 在 RX enqueue 时固化已提交的 `A_N`；decode 入队时记录 clear generation，sole consumer 只丢弃 clear 前报告并保留 clear 后报告；history miss 会作废该 slot 并递降尝试更旧 `M`；TO publish 与 clear 线性化，禁止旧计算重新置 `available=true`；cache 优先淘汰外 epoch、最旧冗余 usable 和最远 future；Alignment 平移旧 UL 谱。
- **CUDA**：与 CPU 对齐绝对帧 eRTM 语义——`CUDAUplinkRxEngine` MATCH 门控测量快照、`CUDAUplinkTxEngine` `state_at(M)` 历史、CUDAUE 冻结 `A_N` / generation clear / force_remeasure 失败作废 TO；延迟谱与相关仍走 CUDA kernels。

## 2026-07-14 - BS-USRP 绝对空口帧序号与 Tier-1 Identity SIB

### Summary

建立以 BS USRP 时间为权威的绝对空口帧序号；每帧注入 **BCH 保护的 Identity SIB**；UE 用 **(BS seq, UE boundary) 时间锚点**推导帧号，并用 **帧中点时间**做稳健相关。TX 硬门控保证波形内 SIB 与空口时隙一致。UL 闭环：UE 将帧序号调制进上行，BS 侧 double-check。

### Changes

- **BS TX 硬门控**：仅当 `claimed_seq == seq_from_ticks(tx_time)` 且 `mod_generation` 匹配才发射；过期丢弃、未来 hold、空槽 blank；underflow 时 `cancel_and_force_next_seq` + 排空队列。
- **原子 claim**：`(mod_generation, seq)` 在 mutex 下快照，避免 cancel 与 claim 交错产生“新 gen + 旧 seq”假 future hold。
- **Boot epoch**：高熵 64-bit boot id（仅 BS 内部时钟）；**canonical OTA epoch32** 写入 `AirFrameIdentity.timeline_epoch` 与 SIB（word0 lo16 + word1 hi16，仍 192 QPSK）。
- **UE 时间锚点**：先 `extend_mod_seq` 再判断异常倒退，避免 32-bit 回绕误触发 timeline reset。
- **resync 安全**：SIB 事件带 `result.generation`；**仅在 publish 与（若需要）ARQ reset 成功后才提交 `last_seen_ota_epoch`**，避免“见过新 epoch 却未 reset”。
- **`last_seen_ota_epoch`（epoch32）** 跨 SYNC_SEARCH 保留。
- **SFO reset 分代**：仅 RX 在成功安装 SIB anchor 后写入 `_pending_sfo_reset_generation`；process 线程仅当它等于当前 `_sync_generation` 时执行；bit 线程只做 ARQ reset。
- **UE 实际采样率**：timeline / boundary / alignment grid phase 统一使用 `get_rx_rate()` 缓存值，避免 UHD coercion 后混用 YAML 速率。
- **BS/UL 容量 fail-fast**：`payload_re_count < kIdentityControlSymbols` 在初始化阶段直接退出。
- **UL 闭环硬门控**：UL period 携带目标 `(epoch, seq, period_generation)`；UL-TX 按 timed grid 与 UE 时间锚推导当前目标，只发送身份和代际完全匹配的 period，过期丢弃、未来 hold、空槽 blank。同步清除和 timed restart 会递增代际，使队列内及 producer 手中的在途旧 period 一并失效；同 epoch 的 claim 落后不 gen++；BS UL-RX 对 `(epoch, seq)` 做精确 double-check。
- **诊断日志**：独立模块 `air_frame`（YAML `logging.modules.air_frame: info`）输出 `[AirFrame]` 心跳；Web Editor Logging 按 BS/UE 过滤无关模块。
- **ARQ reset**：`ArqTxWindow::reset(config, dir)` 持锁；`configure`/`set_direction` 亦加锁。
- **传播**：`RxFrame`/`SensingFrame`/LLR/UL-RX 带 identity；双站感知用 `frame_start_symbol`。
- `ue_frame_mid_time_ns = boundary + T/2` 作为首选稳健时间戳。

## 2026-07-13 - ARQ 协议窗口上限与安全默认值分离

### Summary

ARQ 窗口保留 16 位序列号半空间约束允许的 `32767` 协议上限，同时将 C++ 默认值、配置模板、benchmark 模板和 Web Editor 默认值统一为更保守的 `256`。

### Changes

- `arq_window_packets` 的运行时归一化和 TX/RX 窗口允许范围恢复为 `[1, 32767]`，使显式配置的大窗口不再被强制截断到 `256`。
- 所有 BS/UE、硬件/仿真和 benchmark YAML 模板继续默认使用 `256`，避免默认配置在 ACK 丢失时保留过多原始 payload 与 LDPC/QPSK 编码副本。
- Web Editor 和中英文 YAML 参考文档同步区分“默认值 256”与“协议最大值 32767”。

## 2026-07-13 - Astro 文档路由与维护流程整理

### Summary

文档站工具与工作流栏目使用 `tools-workflows` 路由，中英文内容直接在 Astro/Starlight Markdown 源文件中维护。移除已不再使用的 README 到 Astro 同步脚本及相关工作流说明。

### Changes

- 将“模块与功能”更名为“工具与工作流”，并将中英文路由从 `modules` 调整为 `tools-workflows`。
- 删除 `scripts/sync_docs_from_readme.py`；README 与 Astro 文档站改为独立维护，站点仍通过 `cd site && npm run build` 发布到 `docs/`。
- 新增中英文“参考总览”入口，改写 YAML 使用说明、脚本命令和按现象排列的故障排查流程，并统一中文参数表中的用户向表达。
- `ChannelSimulator` 在 UE→BS 上行重采样回 BS 时钟域后施加反向本振失配，并通过共享控制块消费 UE 上行 TX retune，动态更新上行残余 CFO。TDD 直接复用下行 RX 校正量，FDD 的初始失配和 TX 校正都按上下行载频比缩放。
- 双工仿真模板显式将 BS 上行 RX 窗口偏移和 UE 上行 TX timing advance 设为零，避免继承真实硬件校准默认值而移动仿真直达径。
- 将 latest-only 调试发送队列的正常丢帧与旧帧替换日志降为 `debug`；真正的发送异常仍保留为 `warning`。
- eRTM 新增 `uplink.ertm_timing_metric`，可在原有 `delay_magnitude` 指标与复信道相干相关的 `maximum_likelihood` 指标之间切换；CPU/CUDA UE、配置模板、Web Editor 与中英文文档使用一致的选项语义。
- 仿真 radio backend 改用统一的绝对样本时间轴：共享内存流记录绝对起点，模拟设备时钟随已处理的空中样本推进，TX/RX 执行 timed start，尚未开始的上行区间按空口静默继续推进 BS RX。首个 TX 前时钟保持为零，使不同启动顺序下的 BS/UE 选择相同起点；RX 请求早于实际 TX 原点时会对齐到该原点。UE 下行接收对齐与上行发送窗口因此共享同一时钟参考，且仿真低于实时速度时不会产生墙钟驱动的虚假预测时延。
- 启用仿真 `predictive_delay` 时，UE 会比较 `cfo_hz` 对应的同源晶振 ppm 与 `sample_rate_offset_ppm`；两者不一致时输出配置警告，避免将独立注入的 CFO 错当成采样时钟漂移。
- 新增成对的 `BS_Sim_eRTM.yaml` / `UE_Sim_eRTM.yaml` 专用仿真模板：默认启用最大似然定时指标、eRTM debug 与绝对时延校正，并关闭单站感知、CFO/SRO 和 predictive delay，提供可直接运行的 eRTM 基线。

## 2026-07-12 - 信号处理文档定时与同步模型重构

### Summary

中英文信号处理文档重新划分初始同步、链路内残余 CFO/SFO 补偿与双站感知定时：UE 初始同步独立说明单 ZC、第二 ZC、CFO training field、候选 CFO 和 CP-tail 细化；上下行通信分别承接信道估计与残差跟踪；OTA LoS tracking 和 eRTM 作为 UE 双站感知中二选一的时偏补偿方法。

### Changes

- 以统一的 BS/UE 端时偏符号补充 TDD 上下行信道互易关系，并在 eRTM 中同时给出时延幅度谱和加权最大似然两类定时差指标。
- 将通信路径时延拆分为真实传播时延、固定 RF 群时延和相对于本地解调边界的时变偏差；明确下行 `tau_align` 表示 UE 解调/FFT 边界，并按相邻 DL/UL 参考符号与缓慢变化 TO 的条件推导互易关系。
- 区分 OTA LoS tracking 的 LoS 相对时延输出与 eRTM 保留真实传播时延的输出，明确 eRTM 不需要 LoS 路径及 FDD 路径集合/散射系数差异对可靠性的影响。
- 将时延、距离和多普勒的分辨率与无模糊范围移入单站/双站感知处理章节，并将中文“资源栅格”统一为“资源网格”。

## 2026-07-11 - 运行期丢帧非阻塞化与丢包告警

### Summary

感知与调试输出的生产路径改为有界、非阻塞入队：队列满时丢弃新帧并输出带累计计数的 Warn 日志，不再等待后台发送线程。系统同时补齐通信、感知、恢复、ARQ、控制与可视化链路中的运行期丢包可观测性；正常 shutdown 清理队列保持静默。

### Changes

- 单通道与聚合感知 sender、latest-only 调试 sender 和 viewer 显示队列在满载或合并旧帧时明确告警；ZMQ 数据发布改用 `XPUB_NODROP`，使发送端高水位溢出以非阻塞失败返回并记录 Warn。
- UDP 输入使用完整数据报缓冲并读取 Linux `SO_RXQ_OVFL` 元数据，显式报告内核接收队列溢出、应用缓冲截断和收发失败。
- 补齐 ARQ 窗口跳跃、重排淘汰、最大重传次数、恢复代际淘汰、部分 RX 帧以及 LDPC/eRTM 无效数据的 Warn 级诊断。

## 2026-07-11 - eRTM 同帧感知测量与异步调试输出

### Summary

UE 将 eRTM 作为感知链路中的 TO 测量：CPU 感知帧携带同一接收帧生成的下行信道估计，CUDA UE 则在当前 demod/sensing frame 进入感知处理前完成估计。由此移除独立 eRTM 计算线程、全局“最新下行信道”快照和 pending alignment 补偿，避免 TO 结果与实际被校正的感知帧使用不同的下行坐标。

### Changes

- LDPC 路径仍只识别并投递 BS 上行测量；CPU sensing 线程或 CUDA 当前帧处理阶段消费最新报告，并在新报告或接收对齐后用当前帧下行信道重算 `TO_UE`。Alignment 会强制重算；若 CPU Alignment 感知帧因队列满被丢弃，该请求顺延到下一幅实际进入 sensing 的帧。
- 未启用双基地感知时，CPU UE 仍在主处理路径消费 eRTM 报告，保留 TO 日志与 debug 功能。
- eRTM debug multipart 输出改由有界 `LatestOnly` 后台 sender 发布；感知路径准备拥有数据的 debug frame 后只做非阻塞入队，ZMQ socket 操作和 message 构造不再占用感知计算线程。

## 2026-07-10 - 双向多通道信号处理文档统一

### Summary

中英文信号处理章节按统一符号体系重写：补充 TDD/FDD 上行通信、BS 多通道 ULA 单站感知、UE 双站连续时偏补偿与 eRTM 双向定时关系，并新增处理总览和上行通信页面。全章聚焦连续的信号模型、估计和变换过程，不再沿用早期 SISO、仅下行表述。

## 2026-07-10 - eRTM TO 校正谱连续化

### Summary

eRTM debug 的上行/下行 TO 校正谱复用 Sensing 的 pending alignment 补偿：UE 侧使用 `TO_UE + pending_alignment`，BS 侧使用相反的 pending 项，在整数接收对齐前后保持两幅校正谱连续。

## 2026-07-10 - Config console runtime CPU plan panel

### Summary

`config_web_editor` runtime 面板重构：右侧展示 Isolated / Bound / Process / System CPU 列表与关键角色明细；新增 **Apply Isolation**；Save+Start / Apply / Reset 对齐 `isolate_cpus.py`（关键核 isolate，进程 AllowedCPUs 为全核）。

## 2026-07-10 - YAML-driven CPU isolation (critical cores only)

### Summary

`scripts/isolate_cpus.py` 重写 CPU 隔离：默认先询问本机是 **BS / UE / BS+UE**，再从对应 YAML 提取关键线程核（USRP 收发、OFDM 调制/解调、main；BS 含 sensing RX ingest）。系统 slice 避开这些核；`run` 的进程 `AllowedCPUs` 为全部逻辑核，非关键线程可使用系统核。Web 配置台启动路径同步该策略。

## 2026-07-10 - Replace profiling_modules with hierarchical logging

### Summary

移除 `runtime.profiling_modules`；原 debug 与性能剖析开关统一并入 `logging.modules`。性能相关模块路径以 `_profiling` 结尾，默认关闭且不继承父级 verbosity。

- 全局默认日志级别改为 `warn`，未显式配置的 `info` / `debug` 诊断与 profiling 保持关闭。
- latency、sensing 与仿真 profiling 脚本仅显式启用各自所需的 `_profiling` 模块。

### Mapping (legacy → logging.modules)

| Legacy | Kind | Module path |
|--------|------|-------------|
| demodulation / latency | perf | `demod_profiling` / `cuda.demod_profiling` |
| breakdown_eq | perf | `demod_eq_profiling` / `cuda.demod_eq_profiling` |
| modulation / latency | perf | `mod_profiling` / `cuda.mod_profiling` |
| ldpc_encode | perf | `mod_ldpc_profiling` / `cuda.mod_ldpc_profiling` |
| sensing / sensing_proc | perf | `sensing_profiling` / `cuda.sensing_profiling` |
| udp_egress | perf | `udp_egress_profiling` |
| snr | debug | `demod.snr` |
| agc / arq / ertm / uplink / ue_recovery / sync / cfo / align | debug | `agc` / `arq` / `ertm` / `ul_tx` / `recovery` / `sync` / `cuda.cfo` / … |

## 2026-07-09 - Hierarchical module logging

### Summary

AsyncLogger 增加分级（error/warn/info/debug）与层次化模块过滤（如 `demod.ldpc`、`cuda.demod.ldpc`），YAML `logging:` 与 config_web_editor 树形编辑支持；Error 默认强制输出，仅用于不可恢复失败。

### Changes

- `LogLevel` 扩展 `Debug`/`Off`；输出前缀 `[LEVEL] [module.path]`；可选 timestamps。
- 编译期 `LogModule` 树 + 原子 effective level 表；`LOG_*_M(Module)` 宏；旧 `LOG_G_INFO()` 等挂到 Root。
- `logging.default_level` / `force_error` / `modules` YAML；BS/UE/ChannelSimulator/CUDA 启动后 `apply_logging_config`。
- 高频前缀调用点（Demod/LDPC/UL/eRTM/sensing/CUDA…）改挂对应模块。
- config_web_editor 新增 Logging 分区与 `logging_mapping` 树形 level 选择。

## 2026-07-09 - Uplink self-scan 切片传输与 viewer 简化

### Summary

UE self-scan 诊断改为只发布峰值附近 correlation 切片 + 固定 metadata，显著减小 ZMQ 数据量；viewer 纯显示、不再支持客户端 slice 选择。同步修复 self-scan 默认端口与 eRTM debug 端口冲突导致的 libzmq `!_more` 崩溃。

### Changes

- UE 发布格式：`ULSCSCAN` 64 字节 LE header + `complex64[slice_len]`；默认切片长度为一个带 CP 的 OFDM 符号（`fft_size + cp_length`，配置 `debug_self_scan_slice_samples=0` 表示自动）。
- `scripts/plot_uplink_self_scan.py` 改为解析 metadata/slice 的纯显示端，移除 `--slice` / `--frame-samples` / auto-slice 等客户端裁剪选项。
- `network_output.self_scan_port` 代码默认值改为 `12352`；全部 UE 模板补齐 `self_channel_*` / `self_pdf_*` / `self_scan_*`。
- `scripts/qt_debug_plot_common.py` 不再设置 `ZMQ_CONFLATE`；小 `RCVHWM` + `recv_multipart` 完整收帧。
- `config_web_editor` 增加 `debug_self_scan_slice_samples`；`self_scan_ip/port` 依赖 `debug_self_channel` + `debug_self_scan_spectrum`。

## 2026-07-08 - eRTM RF delay 样本单位配置

### Summary

将 UE eRTM RF 链路校准延迟配置从纳秒单位改为样本单位，便于直接填入单站 system delay estimation 得到的小数采样点结果。

### Changes

- `uplink.ertm_dl_rf_delay_ns` / `uplink.ertm_ul_rf_delay_ns` 重命名为 `uplink.ertm_dl_rf_delay_samples` / `uplink.ertm_ul_rf_delay_samples`，类型保持 `double`，支持小数样本值。
- UE eRTM `tau_c_samples` 现在直接使用配置中的 RF delay samples，不再按 sample rate 从纳秒换算。
- 该重命名不保留旧字段兼容；旧 `*_ns` 字段不会再被读取。

## 2026-07-04 - CUDA eRTM TO 路径

### Summary

将 eRTM TO payload 生成、消费、日志和 `ertm_absolute` 感知时延校正扩展到 CUDA BS/UE 路径。

### Changes

- CUDABS 现在从 CUDA 上行 RX 的频域信道估计缓存最新 uplink channel，并按 `ertm_report_interval_frames` 注入带 mini-header `kFlagErtmTiming` 的内部下行 LDPC payload。
- CUDAUE 现在消费 eRTM timing payload，复用 UE 侧过采样 delay spectrum、相关峰 centroid3 细化、TO 日志和 debug ZMQ 输出格式。
- CUDAUE 的 eRTM 过采样 IFFT、幅度相关、相关峰搜索和 centroid3 邻点采样现在通过专用 CUDA stream、cuFFT 和 CUDA kernels 执行；常规 TO 估计只回传 block energy、block peak 和 centroid 邻点，debug 输出开启时才回传完整可视化谱。
- CUDAUE 的 `sensing_delay_correction_mode=ertm_absolute` 不再回退到 `los_tracking`，在已有 eRTM `TO_UE_samples` 时会用于双站感知 delay correction，并保留 pending alignment 补偿日志。

## 2026-07-02 - eRTM 调试谱输出

### Summary

新增 UE 侧 eRTM debug ZMQ 输出和独立 Python 绘图脚本，用于观察上行/下行 delay spectrum 与 eRTM 相关谱。

### Changes

- BS 在 eRTM payload 中发送 IFFT 前的频域上行信道，UE 端用 BS 上行信道和本地下行信道计算 10 倍补零 IFFT delay spectrum，并基于过采样谱做 eRTM 相关与 TO 样本点估计。
- UE 在 `uplink.ertm_debug_output_enabled=true` 时发布 eRTM debug multipart 包，包含 10 倍过采样的 BS 上行 delay spectrum、UE 下行 delay spectrum、归一化相关谱、相关峰值位置和 TO 样本点元数据。
- eRTM debug ZMQ 包只发送零延迟附近的中心窗口，完整 10 倍过采样谱仍用于 UE 端 eRTM 相关和 TO 计算。
- `ertm_dl_rf_delay_samples` / `ertm_ul_rf_delay_samples` 仅保留在 UE 配置模板和 UE 配置编辑器中，`ertm_debug_output_enabled` 放在这两个 UE 校准项之后。
- eRTM debug 包新增 C++ 频域 TO 补偿后再过采样 IFFT 的上行/下行 delay spectrum，`scripts/plot_ertm_debug.py` 的 `TO correction` 开关仅切换显示 debug corrected 谱，不影响 eRTM 估计计算。
- 新增 `network_output.ertm_debug_ip` / `network_output.ertm_debug_port`，默认绑定 `0.0.0.0:12362`。
- 新增 `scripts/plot_ertm_debug.py`，实时绘制上行 delay spectrum、下行 delay spectrum 和 eRTM 相关谱，并标注相关峰值与 TO 估计。

## 2026-07-02 - eRTM TO 日志单位调整

### Summary

将 eRTM TO 日志输出统一改为采样点单位，并把 RF 链路校准延迟配置改为样本单位。

### Changes

- `UE` 的 `[eRTM] TO` 日志现在打印 `tau_c_samples`、`TO_BS_UE_samples`、`TO_UE_samples` 和 `TO_BS_samples`，保留小数采样点结果。
- eRTM RF delay 配置字段重命名为 `ertm_dl_rf_delay_samples` 和 `ertm_ul_rf_delay_samples`，模板配置、Web 配置编辑器 schema 和 README 参数表同步改为 samples 单位。

## 2026-07-01 - CPU eRTM TO 估计日志

### Summary

新增 CPU BS/UE 路径的 eRTM TO 估计到日志输出阶段，用于对比 BS 上行 delay spectrum 与 UE 下行 delay spectrum，并结合运行时 DUTI/TADV 打印 TO_BS_UE、TO_UE、TO_BS。

### Changes

- BS CPU 下行发送路径在 `uplink.ertm_to_enable=true` 时，将最新上行 delay spectrum、采样率、运行时 `DUTI` 和序号封装为内部 `ERTM1` LDPC payload，并按 `ertm_report_interval_frames` 周期注入到普通 UDP payload 前。
  影响：该 payload 使用普通 LDPC mini-header，UE 通过 payload magic 识别；不改变现有 LDPC framing。

- UE CPU 下行解码路径拦截 `ERTM1` payload，不转发到用户 UDP；它与本地最新下行 delay spectrum 做归一化幅度循环互相关，并用 `ertm_dl_rf_delay_samples`、`ertm_ul_rf_delay_samples`、payload `DUTI` 和本地运行时 `TADV` 打印 TO 估计。
  影响：当前只做估计和日志，不自动修正定时。

- ChannelSimulator 的 UE->BS uplink path 现在复用 BS->UE communication path 的 `comm_multipath_taps` 和双站散射体场景，不再暴露或读取单独的 `uplink_multipath_taps`。
  影响：仿真 TDD 上下行信道形状保持互易，避免 eRTM delay spectrum 对比时由独立上行通道引入人为偏差。

- eRTM payload 注入不再依赖 BS 下行 UDP 输入唤醒，且 `ertm_report_interval_frames` 默认值和配置模板调整为 `32`。
  影响：无用户下行数据时仍可持续输出 eRTM TO 日志；默认实时报告频率约为原 `16` 帧间隔设置的一半。

- ChannelSimulator 新增默认启用的 `simulation.pacing_enabled`，按 `chunk_samples / sample_rate` 将共享内存输出限速到 wall-clock 采样时间。
  影响：sim backend 默认更接近硬件实时推进，eRTM/感知/通信日志频率不再随 CPU 可跑多快而暴涨；需要最快 batch 仿真时可显式关闭该开关。

## 2026-06-29 - 移除旧兼容路径

### Summary

删除旧配置字段和旧接口的兼容路径，收敛到当前 YAML schema 与 ZMQ 控制接口。

### Changes

- BS/UE 配置加载不再从 `network_output` 读取 `arq_enabled`、`arq_ordered_delivery`、`arq_window_packets`、`arq_ack_bitmap_bits`、`arq_retransmit_timeout_ms`、`arq_max_retries` 或 `arq_feedback_interval_ms`。
  影响：旧 ARQ 配置字段不会再悄悄生效；需要把下行 ARQ 参数放到 `downlink` section，上行 ARQ 参数放到 `uplink` section。

- 控制心跳接口改为无参数广播，移除仅用于旧调用点兼容的 `send_heartbeat(ip, port)` 签名、Python viewer `_update_sender()` no-op 和未使用的 `pack_legacy_command()`。
  影响：ZMQ 控制链路继续向已知 peer 广播 RDY；旧 UDP 发送端发现路径不再保留空实现。

## 2026-06-29 - CUDA UE 失锁重同步上行窗口保持

### Summary

修复 CUDAUE normal-stage 失锁进入重搜时清空 UL-TX/RX alignment shift 的问题，使其与 CPU UE 的上行失锁重同步恢复语义一致。

### Changes

- CUDAUE 普通失锁重搜现在保留已收敛的 UL-TX/RX alignment anchor；只有共享 RX/UL-TX 流重启路径仍清空该 shift，因为重启会把 RX 和 UL-TX 一起重新锚定到新的 timed start。
  影响：CUDAUE 失锁后重新同步时，上行 TX 窗口不会回退到 TADV-only 位置，避免 TX-RX 窗口关系被 normal resync 打乱。

## 2026-06-29 - UE 感知线程 CPU 绑核拆分

### Summary

将 UE 的双站感知处理线程从 `downlink_cpu_cores` 中拆出，改为独立的 `sensing_cpu_cores` 配置项；Web 配置编辑器在 `sensing.bi_enabled=false` 时隐藏该字段。

### Changes

- UE / CUDAUE 的下行 CPU 绑核现在只覆盖 `rx_proc`、`process_proc` 和 `bit_processing_proc`，默认 `downlink_cpu_cores: [1, 2, 3]`；`sensing_process_proc` 改由 `sensing_cpu_cores[0]` 绑定，默认 `sensing_cpu_cores: [4]`。
  影响：关闭双站感知时不再显示或需要配置感知处理核；开启时可单独给 sensing 线程分配 CPU，不再占用 downlink 列表槽位。

## 2026-06-29 - CUDA UE 上行收发共享定时启动

### Summary

修复 CUDAUE 上行在真实硬件上不可见（BS `self_pdf` 无信号，而 CPU `UE` 正常）的根因：BS `UplinkRxEngine` 对上行不做 sync/相关搜索，只按固定窗口（`ul_sample_offset` + 被动 `bs_dl_ul_timing_diff`）提取上行，因此 UE 的 UL-TX 帧栅格必须和它的 RX/DL 帧栅格锁定到同一绝对时间锚点。CPU `UE` 用同一个 `stream_start_time` 同时定时启动 RX 流和 `set_timed_tx`；CUDAUE 之前 TX 定时在 `ceil(now+1)` 而 RX 用 `STREAM_MODE_START_CONTINUOUS`（立即启动），两个锚点相差约 1 秒（帧周期内近似随机相位），超出 `rx_alignment_shift` 细对齐能补偿的范围，上行落不进 BS 固定窗口。

### Changes

- CUDAUE 新增 `_next_timed_stream_start()`，在 `start()` 计算单一 `stream_start_time` 并同时用于 `set_timed_tx` 和 `rx_proc`；`rx_proc` 改为按该时刻定时启动 RX 流（`stream_start_time<=0`，即仿真路径，才用 stream_now 立即启动）。
  影响：硬件 duplex 下 UE 的 UL-TX 帧栅格与 RX/DL 帧栅格共享绝对时间锚点，上行重新落入 BS 的确定性 UL 窗口，`self_pdf` 恢复信号；仿真路径（采样同步）行为不变。

- 共享流重启路径同步改造：overflow / UL-TX async error 触发重启时，重新计算一个共享 `stream_start_time` 并同时用于 RX 定时启动和 `reschedule_timed_tx`，保证重启后上行仍与下行栅格对齐。

## 2026-06-29 - CUDA UE 接收溢出帧丢弃

### Summary

定位并修复 CUDA UE 在真实硬件上 normal-stage pilot CFO 间歇性飞到 -5~-6 kHz 的根因：RX recv 循环未检查 `md.error_code`，USRP overflow 时把溢出前后时间不连续的样本拼接进同一帧，跨越间断点的 pilot pair 产生大相位跳变，被 CFO 估计读成大 residual。CPU UE 一直有 overflow 处理（丢弃残帧并重启共享流），CUDA UE 此前完全缺失。

### Changes

- CUDA UE 的 `handle_sync_search` / `handle_alignment` / `handle_normal_rx` recv 循环现在调用 `_handle_rx_metadata_error` 检查每次 recv 的元数据，遇到 overflow/late/broken-chain/alignment/bad-packet 时中断当前帧填充，并只提交/入队完整连续的帧（短读帧释放回池）。
  影响：CUDA UE 不再 demod 跨 overflow 间断点的帧，消除 bimodal 的 -5~-6 kHz 假 residual CFO 观测；`_accept_normal_cfo_observation` 门限退化为纵深防御而非主修复。

- CUDA UE 补齐 CPU UE 的共享 RX/UL-TX 流重启：RX 线程检测到 overflow 或 UL-TX async error 计数变化后停流、按租约时间重排 `CUDAUplinkTxEngine` 定时发送、执行 RX 侧状态复位（bump `_sync_generation`、清 alignment/计数、关 UL 波形、回到 SYNC_SEARCH）并重启流。
  影响：硬件 duplex 下 overflow 后能恢复共享流时序；GPU pipeline 缓冲仍由 process 线程经 generation 检查丢弃在途帧，重启路径不跨线程触碰 GPU 缓冲。

## 2026-06-28 - CUDA UE CFO 与对齐跟踪

### Summary

本次更新修复 CUDA UE 后续 CFO/SFO tracking 在 TDD uplink/guard 边界附近使用污染 pilot pair 的问题，并补齐 CUDA UE 与 CPU UE 的接收对齐和上行发送时序跟踪行为。

### Changes

- CUDA UE 的 GPU CFO/SFO 估计现在复用 CPU UE 的 TDD CFO tracking skip-mask 规则，跳过 uplink/guard 符号及其相邻符号后再累积相邻 pilot phase。
  影响：初始大 CFO 捕获并调谐后，CUDA UE 的 residual CFO 跟踪不再把 TDD 边界附近的下行/上行过渡符号纳入平均，降低 `Average CFO` 在大 CFO 场景下突然飞到数百或数千 Hz 的风险。

- CUDA UE 的 CFO/SFO phase-diff kernel 改为扫描整帧 clean adjacent downlink symbol pair，并在没有合法 pair 时像 CPU UE 一样跳过频率控制更新。
  影响：复杂 TDD 资源图或 pilot 资源不足时不会把 0 Hz 或无效回归结果当作有效 CFO 观测喂给软件调谐或硬件同步。

- CUDA UE 接收对齐现在补齐 CPU UE 的上行时序联动语义，同时保留 CUDA pipeline 的 alignment gate；同步处理负向 alignment、重置时清空 UL-TX/RX alignment shift，并把 gated fresh sync / residual delay 调整同步到 `CUDAUplinkTxEngine::rx_alignment_shift()`。
  影响：TDD CUDA UE 的上行 TX 窗口会跟随下行 RX frame anchor 变化，避免下行重新对齐后上行仍停留在旧时序。

- CUDA CFO/SFO kernel 现在按 CPU 的 `data_symbol_to_actual_symbol` 顺序筛选 physically adjacent data-symbol pair，并让 CUDA UE 负向 alignment 与 CPU 一样先清零帧缓冲再拷贝短读数据。
  影响：避免 CUDA residual CFO tracking 使用和 CPU 不同的 symbol-pair 集合，同时避免负向对齐时把 RX frame pool 中的旧样本带入 FFT。

- CUDA UE 增加 `profiling_modules: cfo` 诊断模式，低频在同一 RX frame 上运行 CPU mirror CFO estimator 并打印 GPU/CPU CFO、alpha、beta 对比。
  影响：可以直接区分 CUDA kernel/phase branch 问题和 retune 后 stale/control 观测问题，默认不启用且不影响正常实时路径。

- CUDA UE normal CFO 控制现在只累计通过 frequency-adjust gate 的新帧观测，并对 residual tracking 的离群/跳 branch CFO 样本做门限拒绝；每次 RX retune 后重置 434-frame 平均窗口。
  影响：避免 retune 前后的 stale frame 或 pilot phase branch slip 污染下一轮平均 CFO，降低 residual CFO 追踪进入正反馈的风险。

## 2026-06-28 - 信道仿真器采样率偏差

### Summary

本次更新为 ChannelSimulator 增加 UE 相对 BS 的 `sample_rate_offset_ppm` 采样时钟偏差仿真。

### Changes

- 新增 `simulation.sample_rate_offset_ppm` 配置项，并同步到 BS/UE sim 模板、Web 配置编辑器 schema 和信道仿真器文档。
  影响：默认 `0` 保持原有仿真行为；设置非零值时可在不修改 BS/UE 标称 `sample_rate` 的情况下引入信号级采样率偏差。

- ChannelSimulator 在 BS->UE communication path 上按 UE/BS 采样率 ratio 重采样，在 UE->BS uplink path 上按 reciprocal ratio 重采样。
  影响：BS 端 TX、uplink RX 和所有 monostatic sensing RX 通道保持同一个 BS 采样时钟；UE 端 downlink RX 和 uplink TX 保持同一个 UE 采样时钟。

## 2026-06-26 - 空口 ARQ 丢包重传

### Summary

本次更新为上下行 LDPC 空口 packet 增加可选 ARQ 重传机制，并提供保序交付开关。

### Changes

- 新增 `network_output.arq_enabled`、`arq_ordered_delivery`、`arq_window_packets`、`arq_ack_bitmap_bits`、`arq_retransmit_timeout_ms`、`arq_max_retries` 和 `arq_feedback_interval_ms` 配置项，并同步到 BS/UE 配置模板、benchmark 模板和 Web 配置编辑器 schema。
  影响：默认 `arq_enabled: false` 保持原有低延迟 UDP 转发行为；开启后链路层会用 ACK bitmap 反馈确认并重传未确认空口 packet。

- 将下行 ARQ YAML 主配置迁移到 `downlink.arq_*`，BS 端仅暴露发送端窗口/RTO/最大重传次数，UE 端仅暴露接收端保序/反馈间隔参数；旧 `network_output.arq_*` 读取兼容已在后续更新中移除。
  影响：新配置模板为下行 ARQ 预留与后续上行 ARQ 分离的命名空间；默认 RTO 调整为 `100 ms`、最大重传次数为 `5`、ACK feedback 间隔为 `10 ms`。

- 恢复上行 ARQ 并改为由 `uplink.arq_*` 单独控制，BS 端暴露上行接收/ACK 参数，UE 端暴露上行发送窗口/RTO/最大重传次数。
  影响：可以独立开启上行或下行 ARQ；下行 ARQ 仍通过 UE uplink 发送 ACK feedback，上行 ARQ 通过 BS downlink 发送 ACK feedback。

- BS/UE 与 CUDA BS/UE 路径接入共享 ARQ helper，支持重复包抑制、ACK feedback 内部消费、重传窗口和可选保序交付。
  影响：视频等需要顺序输出的场景可以开启 `arq_ordered_delivery`，普通低延迟场景可保持关闭以便收到非重复 packet 后立即转发。

- ARQ feedback 使用 LDPC mini-header flag 标识，并使用独立 feedback sequence space；UE 侧 feedback 注入改为由 uplink encode 线程统一入队。
  影响：feedback packet 不再占用数据 sequence、不会卡住接收窗口或保序交付，也避免了 UE uplink SPSC 队列的双生产者竞态。

## 2026-06-25 - UDP egress pacer 统计日志

### Summary

本次更新为 UDP egress pacer 增加周期性运行统计，方便观察自动检测速率和队列状态。

### Changes

- pacer 启用并有流量时，每约 1 秒打印 `UDP egress pacer stats`，包含 fixed/auto 模式、有效 pacing 速率、自动估计速率、入队/发送 Mbps、队列深度、累计发送包数以及丢包/发送失败计数。
  影响：调试 decoded payload UDP 输出突发、排队和丢包时，可以直接从 BS/UE 日志判断 pacer 是否按预期估计和限速。

## 2026-06-24 - YAML 嵌套字段命名收敛

### Summary

本次更新将外部 YAML 配置字段收敛到各 section 内的短字段名，并移除对应旧字段的前向兼容读取。

### Changes

- `uplink`、`sensing`、`downlink` 下的重复前缀字段改为 section-scoped 短字段名，例如 `uplink.enabled`、`uplink.rx_channel`、`sensing.rx_channels`、`sensing.output_mode`、`sensing.mask_blocks` 和 `downlink.rx_wire_format`。
  影响：新配置不再接受旧的前缀式字段名，模板、Web 编辑器、benchmark 模板和文档均已同步到新 YAML schema。

- Web 配置编辑器新增内部 `data_key` 区分同名字段，使不同 section 可以各自使用 `rx_wire_format`、`rx_channel` 等短 YAML key，同时保持表单数据不会互相覆盖。
  影响：Web 端保存的 YAML 与运行时读取的嵌套 schema 保持一致。

## 2026-06-23 - BS 上行 RX 通道配置生效

### Summary

本次更新修复 BS 侧 `uplink_rx_channel` 配置未接入运行时的问题。

### Changes

- 新增运行时 `uplink_rx_channel` 配置字段，并让 BS 上行 RX 的 tune、gain、bandwidth、stream 和 AGC 都使用该字段；旧配置省略该字段时仍回退到 `rx_channel`。
  影响：BS 可独立设置上行 RX 通道，例如 `uplink_rx_channel: 1` 会真正使用 USRP RX channel 1。

## 2026-06-23 - FDD Uplink 默认载波

### Summary

本次更新将 FDD 上行载波默认值设为 2.5 GHz。

### Changes

- `uplink.center_freq` 默认改为 `2500000000` Hz，并同步到 C++ 默认配置、Web 配置编辑器 schema 和 README。
  影响：新建或补全 FDD 上行配置时，默认 UE->BS 上行载波为 2.5 GHz；TDD 仍忽略该字段并使用下行中心频率。

## 2026-06-22 - UE 上行失锁重同步恢复

### Summary

本次更新修复 UE 上行开启后失锁再重同步时 UL-TX 窗口恢复不稳定的问题。

### Changes

- UE 普通失锁进入重搜时不再清空 UL-TX 的 RX-alignment shift，保留已对齐的上行发送 anchor；重同步成功后 fresh sync acquisition 会强制调度 RX alignment，并通过 UL-TX 多/少发样本继续跟随新的下行 frame anchor。
  影响：BS 重启或大 CFO 失锁后，UE 的上行发送窗口能够随重新同步后的 RX/downlink frame anchor 恢复，而不是回到 TADV-only 位置。

- UE 失锁重搜期间会将 UL TX gain 拉到硬件最小值，重新检测到同步、调度 alignment 或重新开启 uplink waveform 时恢复到配置的 TX gain。
  影响：重搜期间降低本机 UL 残留对下行重同步的干扰，同时恢复同步后自动恢复上行发射功率。

## 2026-06-22 - Uplink 自发自收信道调试

### Summary

本次更新为 BS/UE 上行调试增加本机发送 ZC 的 self-channel 估计，用于辅助 DUTI/TADV 调整。

### Changes

- 新增 `uplink.debug_self_channel` 配置开关。开启后，BS 会在上行 RX 接收窗内按理想 BS 本机下行 ZC 位置截取；UE 会按理想 UE 本机上行 ZC 位置截取，然后分别估计自发自收信道和 delay profile。
  影响：调 DUTI/TADV 时可同时观察对端链路和本机理想时序泄漏参考。

- self-channel delay profile 的校准准则：UE 侧调 `ue_timing_advance` / `TADV`，使 UE self_pdf 峰值落在 `desired_peak_pos`；BS 侧调 `bs_dl_ul_timing_diff` / `DUTI`，使 BS self_pdf 峰值落在 `desired_peak_pos`。
  影响：DUTI 和 TADV 分别用本机自发自收的 ZC 泄漏峰作为参考，避免混用对端链路峰值。

- 新增 self debug 输出端口：BS 默认 `self_channel_port=12360`、`self_pdf_port=12361`，UE 默认 `self_channel_port=12350`、`self_pdf_port=12351`。
  影响：BS/UE 可在同一台仿真主机上同时开启 self-channel debug，端口不会与现有 uplink channel/pdf 或 UE downlink channel/pdf 冲突。

## 2026-06-22 - B210 Duplex 默认配置模板

### Summary

本次更新把当前验证中的 B210 BS/UE TDD duplex 运行配置固化为可复用模板。

### Changes

- 新增 `config/BS_B210_Duplex.yaml` 和 `config/UE_B210_Duplex.yaml`，内容来自当前 `build/BS.yaml` 与 `build/UE.yaml`，包含 B210 UHD 参数、TDD uplink 窗口、idle random QPSK、uplink debug 输出端口和 CPU 绑定默认值。
  影响：B210 双工实验可直接复制 Duplex 模板到 `build/BS.yaml` / `build/UE.yaml`，不必从普通 B210 下行模板手动补上上行参数。

- `bs_dl_ul_timing_diff` 和 `ue_timing_advance` 默认值统一为 `63` samples，并同步到 Web 配置编辑器 schema、B210 模板和 README。
  影响：BS 上行 RX 窗口和 UE 上行 TX timing advance 的启动默认值一致，仍可运行时通过 `DUTI` / `TADV` 微调。

## 2026-06-22 - Uplink TDD 默认窗口调整

### Summary

本次更新将 TDD uplink 默认窗口改为帧尾 10 个 OFDM 符号，并默认保留 1 个 guard symbol。

### Changes

- `uplink.symbol_start` 默认改为 `90`，`uplink.symbol_count` 默认改为 `10`，`uplink.guard_symbols` 默认改为 `1`，并同步到 C++ 默认配置、Web 配置编辑器 schema、配置模板、benchmark 模板和 README。
  影响：新建或补全配置时，默认 TDD 上行窗口为 `[90, 100)`，其中符号 `90` 为 guard，实际上行数据符号为 `[91, 100)`。

## 2026-06-21 - 配置模板与 Web 编辑器分组整理

### Summary

本次更新整理 YAML 配置模板和 `config_web_editor` 的分组显示，使 CUDA、上行、下行和仿真参数在模板与 Web 界面中保持一致。

### Changes

- Web 配置编辑器 schema 将 CUDA 相关参数集中到 `CUDA` 分组，将原 `Duplex` 分组更名为 `Uplink`，并把均衡/信道跟踪、`uplink_rx_*` 上行 RX 模板参数和 uplink CPU 参数归入 `Uplink`。
  影响：上行相关参数不再散落在 OFDM/RF/Device 分组里，查找和保存后的 YAML 顺序更稳定。

- BS 模板中保留 `rx_device_args`、`rx_clock_source`、`rx_time_source` 作为 sensing RX 默认配置，并移入 `Sensing`；新增 `uplink_rx_channel`、`uplink_rx_device_args`、`uplink_rx_clock_source`、`uplink_rx_time_source` 作为上行 RX 模板字段。
  影响：sensing RX 默认设备配置和 Uplink RX 模板配置不再共用同一组字段名。

- 拆分并移除旧的顶层 RX wire-format 配置：BS 使用 `uplink_rx_wire_format` 与 `sensing_rx_wire_format`，UE 使用 `downlink_rx_wire_format`，sensing 每通道覆盖字段改为 `wire_format`。
  影响：上行、感知和 UE 下行 RX wire format 不再共用一个含义模糊的顶层字段。

- `config_web_editor` 的 `uplink` 嵌套字段按 BS/UE 角色拆分：BS 只显示和保存 decoded uplink payload 输出目的地，UE 只显示和保存 uplink payload UDP 输入绑定。
  影响：UE 配置不会再出现 BS 专用的 `uplink` 输出字段，BS 配置也不会再出现 UE 专用的 `uplink` 输入字段。

- `tx_*`、TX wire format、BS 下行 pipeline 和 downlink CPU 配置归入 `Downlink` 分组。
  影响：下行发送路径参数与上行接收/调试参数分离。

- `simulation` 在 Web 编辑器中仅当 `radio_backend=sim` 时显示，并改为 `simulation.xxx` 扁平行显示。
  影响：UHD 配置下不会显示仿真专用参数，仿真配置编辑时也减少嵌套层级。

- `radio_backend=sim` 时，Web 编辑器隐藏 USRP device/clock/wire format、TX/RX gain/channel、硬件 AGC、硬件同步和 sensing RX 每通道硬件字段。
  影响：仿真配置界面只保留仿真和仍会影响仿真运行的参数；隐藏字段仍保留在 YAML 数据中，切回 UHD 后不会丢失。

- Web 编辑器新增可选字段依赖显示：例如 `cfo_training_period_samples` 仅在 `enable_cfo_training_sequence=true` 时显示；Uplink 细项仅在 `enable_uplink=true` 时显示；measurement/AGC/hardware-sync/AKF/bistatic-sensing 等细项也会跟随对应 enable 开关显示。
  影响：常用配置界面更简洁，关闭功能时不会继续展示只在该功能开启后才生效的细项。

- `scripts/config_web_editor.py --standardize-configs` 新增模板标准化入口，并已用该入口重写 `config/*.yaml`。
  影响：后续可重复用同一套 schema 顺序批量整理配置模板。

## 2026-06-21 - BS 上行调试输出

### Summary

本次更新为 BS 侧 UE->BS uplink 接收链路增加信道估计、delay profile 和星座图 ZeroMQ 调试输出。

### Changes

- `UplinkRxEngine` 在完成上行同步符号信道估计和均衡后，会输出上行 `H_est`、由 `H_est` 生成的 delay profile，以及抽样后的均衡星座符号。
  影响：调试 UE 上行链路时，可以像观察 UE 下行接收一样查看 BS 侧上行信道、PDP/PDF 和星座质量。

- BS 配置、Web 配置编辑器 schema 和配置模板新增 `uplink_channel_ip` / `uplink_channel_port`、`uplink_pdf_ip` / `uplink_pdf_port`、`uplink_constellation_ip` / `uplink_constellation_port`，默认端口分别为 `12358`、`12359` 和 `12356`。
  影响：BS 上行调试流默认避开 UE 下行调试流的 `12348` / `12349` / `12346`，便于 BS/UE 在同一机器上同时运行。

- `scripts/plot_channel.py`、`scripts/plot_pdf.py` 和 `scripts/plot_const.py` 底部连接输入框支持 `host:port` / `tcp://host:port`，例如 `127.0.0.1:12358`。
  影响：查看 BS 上行信道、delay profile 和星座图时，可以直接在窗口输入框里切到 `12358` / `12359` / `12356`，不需要重启脚本或使用额外命令行参数。

- 新增 `scripts/uplink_timing_control.py` PyQt6 调参面板，可分别连接 BS 和 UE 的控制 IP/端口，并用 `-10` / `-1` / `+1` / `+10` 按钮微调 `DUTI` 和 `TADV`。
  影响：BS 与 UE 位于不同主机时，可以在同一个窗口里分别向两端发送上行定时调整命令。

- BS 侧 `UplinkRxEngine` 的 RX stream 启动改为使用 BS 统一 `_start_time` 的 timed start，而不是 `stream_now=true`。
  影响：UL-RX 的帧 0 与 BS TX 的帧 0 共用同一个 radio-clock 锚点，`DUTI` 只负责在该锚点基础上微调窗口偏移，避免启动线程时刻决定上行接收窗口。

- UE 侧启用 uplink 时，RX stream 和 UL-TX timed TX 改为使用同一个未来整秒 radio-clock 启动锚点。
  影响：UE 的下行接收与上行发射在启动时共享同一个硬件时钟起点，避免 RX immediate start 与 TX `now+1s` 各自启动导致帧边界初始偏移。

- UE 侧增加共享 RX/UL-TX stream 重启恢复：RX overflow / metadata error 或 UL-TX underflow / async timing error 发生时，重新调度同一个未来整秒启动锚点，并将接收状态回到 sync search。
  影响：异常恢复时 UE 的下行 RX 与上行 TX 不会各自独立重启导致帧边界再次错开。

- UE 侧 UL-TX 的 `TADV + RX alignment` 调整改为通过连续发送流里的样本数变化生效：提前时缩短当前 period，推迟时补零延长当前 period；每帧 `md.time_spec` 也按目标发送窗口同步移动。
  影响：连续发送时不只依赖 USRP 对每帧 metadata time 的重新定时，后续上行帧边界会同时通过实际样本流移动。

- 上行子帧改用独立 ZC root，并在 UE UL-TX / BS UL-RX 启动日志中打印实际 `zc_root`。
  影响：上行同步符号不再和下行同步 root 或 sensing-pilot alternate root 重复，避免 UE 下行同步搜索误锁到自己的上行信号。

- BS 侧 TX underflow / time-error 触发 burst restart 时，会把新的 BS TX frame anchor 传给 `UplinkRxEngine`；UL-RX 自身遇到 RX metadata/overflow error 时也会按下一帧边界重启。
  影响：UL-RX 会 stop/start RX stream，并按 timed anchor + DUTI 重新对齐，避免 BS TX 与 UL-RX 在异常恢复后停留在不同帧边界。

- UE 侧 UL-TX 在下行同步完成前会把整个上行 waveform 置零，完成初始 alignment 进入正常接收后才恢复发送；如果后续 reset 回到 sync search，会再次静音上行。
  影响：打开 uplink 时，UE 下行同步搜索和初始 CFO 估计不会在未锁定阶段先看到自己的上行泄漏，降低自发射导致的误锁、CFO 偏移和 `No valid delay found` reset。

- UE 侧 TDD 下行 CFO/SFO 估计只使用实际相邻且不贴近 guard/uplink 边界的下行 pilot symbol pair。
  影响：打开 uplink 后，跨过 TDD 上行窗口的符号间隔不会被误当作一个 OFDM symbol 的相位差来调 RX 频率，降低 UL-TX 泄漏对 CFO tracking 的干扰。

- UE 侧调度 RX alignment 时同步更新 UL-TX 的连续发送边界，并将 RX stream-read delta 反号映射到 UL-TX timing target。
  影响：初始对齐和 NORMAL 状态下的小步残余时延调整都会提前通知 UL-TX 在后续 period 中补零/截短，减少 TX 因等待 RX alignment 完成而晚一帧生效的概率；推迟方向也会在日志中显示 `lengthening` 调整。

- UE 侧软件 CFO 校正更新 RX DSP tune 时，同步把对应目标频率 correction 应用到 UL-TX tune；由于 UHD 的 TX DSP 符号与 RX 相反，UL-TX 写入的 `dsp_freq` 使用反号。
  影响：TDD 下 UE 上行发射会跟随下行接收的 CFO 校正保持同载频参考；FDD 下按 UL/DL 载频比例缩放目标 correction，避免上行仍停留在未校正频点。

## 2026-06-21 - Duplex uplink 总开关语义修正

### Summary

本次更新将顶层 `enable_uplink` 接入 BS/UE 的实际上行链路控制，并与 `simulation.enable_uplink` 的仿真通道控制解耦。

### Changes

- 顶层 `enable_uplink` 现在控制 BS/UE 是否创建 uplink TX/RX engine，以及 TDD 模式是否为上行窗口 blank 下行符号；`simulation.enable_uplink` 仅控制 ChannelSimulator 是否创建并转发 `ul.tx` / `rx.ul` 仿真通道。
  影响：关闭顶层 `enable_uplink` 后，BS/UE 会保持全下行帧，不再打开上行链路或保留 TDD uplink 窗口；仿真 hub 的上行通道仍可独立配置。

- 下行 data-resource layout 会跳过 TDD guard/uplink 符号，并忽略落在上行窗口内的 mid-frame pilot；BS 预生成符号模板时也使用实际下行 data-symbol 映射。
  影响：TDD 上行窗口不会再被 UE 当作下行星座/LLR 数据或信道跟踪 pilot，避免星座图显示空白上行符号上的噪声。

## 2026-06-21 - UE 上行 TX wire format 配置

### Summary

本次更新为 UE 配置增加 `wire_format_tx`，用于控制可选 UE->BS uplink TX stream 的 UHD on-the-wire 数据格式。

### Changes

- UE YAML 读写、配置模板、benchmark 模板、Web 配置编辑器 schema 和 README 参数表新增 `wire_format_tx`，默认值为 `sc16`，可选 `sc16` / `sc8`。
  影响：启用 duplex/uplink 时，UE 侧 TX wire format 可以与 RX wire format 分别配置，不再只能依赖代码默认值。

## 2026-06-21 - UE 上行 idle 随机 QPSK 波形

### Summary

本次更新为 UE->BS uplink 增加可配置 idle 波形。UE 没有上行 UDP payload 时，默认仍发送合法 zero-length mini-header，并用确定性随机 QPSK 填充剩余 payload RE。

### Changes

- 新增 `uplink_idle_waveform` 配置，默认 `random_qpsk`，可选 `random_qpsk` / `zero`，并接入 Web 配置编辑器 schema、README 和默认 YAML；`zero` 模式发送 zero-length header，剩余 payload RE 保持为 0。
  影响：没有上行业务数据时，UE uplink 可保持接近真实数据的 QPSK 占用，同时 BS 通过 zero-length header 不会输出 UDP payload。

- `UplinkTxEngine` 复用现有 `splitmix32()` deterministic random helper 生成 idle QPSK，不新增独立随机生成器。
  影响：idle 填充可重复、轻量，并保持与现有 OFDM 随机参考序列实现风格一致。

## 2026-06-21 - Web 配置编辑器 TDD uplink 资源保护

### Summary

本次更新改进 Web 配置编辑器的 duplex/uplink 配置和资源图显示，让 TDD uplink 符号与下行同步、CFO training 和感知/通信资源块的关系更清楚。

### Changes

- `enable_uplink`、`duplex_mode` 和 `uplink.*` 统一放入 Web 配置编辑器的 Duplex card，`uplink` 不再作为 raw YAML/JSON 文本编辑，而是直接展开为 `symbol_start`、`symbol_count`、`guard_symbols`、`center_freq` 和 uplink UDP 输入/输出子字段。
  影响：配置 TDD/FDD uplink 时更容易保持字段类型和单位正确，`center_freq` 继续按 GHz 显示、按 Hz 写入。

- Resource Map 和 Sensing Resource Map 会标出 TDD guard/uplink 符号，并在 TDD 范围覆盖同步符号、第二同步符号或 CFO training field 时给出冲突提示。
  影响：运行前即可看到会导致 TDD/DL 保留字段冲突的 symbol window。

- 保存配置时，`data_resource_blocks` 和 `sensing_mask_blocks` 会自动裁掉落入 TDD guard/uplink 符号的部分；compact sensing 仍会裁掉 CFO training field。
  影响：感知 pilot、payload 和 compact sensing mask 不会继续占用 TDD uplink 符号。

- BS 侧 `UplinkRxEngine` 拆分为 RX sample ingest、OFDM/LLR signal processing、LDPC decode + UDP output 三个线程，并对应 `uplink_cpu_cores[0..2]`。
  影响：上行接收链路可以像 UE 接收链一样分别绑定采样、信号处理和解码/输出线程。

## 2026-06-20 - BS/UE TDD/FDD duplex uplink 支持

### Summary

本次更新把原来的单向 BS->UE OFDM 链路扩展为可选的 UE->BS duplex uplink。TDD 模式在同一载波和帧周期内为上行预留符号窗口；FDD 模式在独立上行载波上连续发送，同时保留下行载波。

### Changes

- 新增 `enable_uplink`、`duplex_mode`、`uplink.symbol_start`、`uplink.symbol_count`、`uplink.guard_symbols` 和 `uplink.center_freq` 等配置，并在 BS/UE 启动时生成一致的 `DuplexFrameLayout`。
  影响：同一套 YAML 可以描述下行-only、TDD uplink 和 FDD uplink 三种运行方式；TDD 会按符号粒度划分 DL、guard 和 UL 数据窗口。

- TDD 模式下，BS 下行调制会在 guard/uplink 符号内 blank 下行资源，UE 只在配置的上行数据窗口内放置自包含 uplink OFDM 子帧；FDD 模式下，下行持续发送，上行使用 `uplink.center_freq`。
  影响：TDD 不会在同一符号内同时占用 DL 和 UL；FDD 可在硬件和隔离条件满足时保持双向链路连续。

- 新增 UE 侧 `UplinkTxEngine` 和 BS 侧 `UplinkRxEngine`。UE 从 `uplink.udp_input_ip` / `uplink.udp_input_port` 接收上行业务，完成 LDPC/QPSK/OFDM 调制并发送；BS 提取上行窗口，完成同步符号信道估计、均衡、LLR、LDPC 解码，并从 `uplink.udp_output_ip` / `uplink.udp_output_port` 输出解码 payload。
  影响：平台新增真正的 UE->BS 数据路径，而不是只在下行链路上做回环或配置占位。

- 上行子帧复用下行 numerology 和 comb pilot 设置，使用不同于下行同步 root 的独立 ZC root，并去掉第二同步符号、CFO training field、mid-frame pilot、sensing pilot 和自定义下行资源块。
  影响：上行解调可以复用现有 OFDM/LDPC 基础设施，同时避免把下行专用字段误带入上行子帧。

- 新增 BS 侧 `bs_dl_ul_timing_diff` / `DUTI` 和 UE 侧 `ue_timing_advance` / `TADV` 控制，用于在共享无线时钟下调整 BS 上行接收窗口和 UE 上行发送窗口。
  影响：TDD 上行收发切换和硬件/链路延迟补偿可以在运行时微调，不需要重新生成配置文件。

- ChannelSimulator 新增 uplink transport，将 UE `ul.tx` 送到 BS `rx.ul`；仿真配置支持 `simulation.enable_uplink`。
  影响：TDD/FDD duplex uplink 可以先在仿真链路中做双向 packet 验证，再迁移到 USRP 硬件。

- CMake、配置模板、benchmark 脚本、Web 配置编辑器和 README 从 `OFDMModulator` / `OFDMDemodulator` 命名迁移到 `BS` / `UE`，并将运行时 YAML 命名为 `BS.yaml` / `UE.yaml`。
  影响：配置和工具命名与 duplex 角色一致，后续描述上行/下行路径时不再混用调制器/解调器概念。

## 2026-06-19 - ZeroMQ debug 监听地址默认值修正

### Summary

本次更新修正了解调端 ZeroMQ debug PUB 输出的默认监听地址语义，并让简单 Matplotlib debug viewer 支持运行时切换后端 Host。

### Changes

- `channel_ip`、`pdf_ip` 和 `constellation_ip` 默认改为 `0.0.0.0`，字段留空时也解析为 `0.0.0.0`，不再继承 `default_out_ip`。
  影响：ZeroMQ PUB socket 不会再尝试绑定远端目标地址；远端 viewer 应连接后端机器的实际网卡 IP。

- 更新 Demodulator 示例配置、benchmark 模板、README、网页文档和 Web 配置编辑器 schema。
  影响：模板和配置 UI 都明确区分 ZeroMQ 监听地址与 UDP 目标地址。

- 为 `plot_channel.py`、`plot_pdf.py` 和 `plot_const.py` 增加 Host 输入框与 Connect 按钮。
  影响：这些简单 debug viewer 可以在窗口内切换连接的后端地址，不需要手动修改脚本常量。

## 2026-06-19 - ChannelSimulator 在线 SNR 控制

### Summary

本次更新为 ChannelSimulator 增加独立 ZeroMQ 控制端口和在线目标 SNR 调整能力，并提供专用 Python 控制界面。

### Changes

- 新增 `simulation.snr_control_enable`、`simulation.target_snr_db` 和 `simulation.control_port` 配置。
  影响：仿真链路可以在 AWGN 噪声功率固定的情况下，通过缩放加噪前的有效信号实时改变目标 SNR。

- 新增 `scripts/channel_sim_control.py`。
  影响：开发者可以通过 PyQt 界面连接 ChannelSimulator 的 ZMQ 控制端口，查询/设置目标 SNR 或关闭目标 SNR 缩放。

## 2026-06-13 - CFO 训练、viewer 重构与配置编辑器增强

### Summary

本次更新在 V1.1 的仿真/ZeroMQ 基础上继续收敛同步、感知显示和配置编辑工作流。重点是加入第二同步符号和专用 CFO training field、改进接收端 CFO alias 解析，整理 fast viewer 架构，并让配置编辑器更清楚地显示资源映射。

### Changes

- 新增 `enable_sec_sync_symbol`，可将 `sync_pos-1` 预留为重复 ZC 第二同步符号。
  影响：接收端可以利用两个连续同步符号做类 Schmidl-Cox 粗定时/模糊 CFO 估计，再通过局部 ZC 相关解析 CFO alias 并精修主同步。

- 新增 `enable_cfo_training_sequence` 与 `cfo_training_period_samples`，可将 `sync_pos+1` 预留为专用重复 CFO training field。
  影响：接收端可以在 ZC/第二同步符号完成帧定位和 modulo CFO 估计后，用专用训练字段进一步解析 CFO alias，提升大 CFO 场景下的同步稳定性。

- 改进同步 CFO acquisition、alias search 范围配置和仿真反馈，并在 README、配置模板和 Web schema 中补齐相关字段。
  影响：同步调试时能看到更明确的 alias 选择依据，TX/RX 配置也更容易保持一致。

- 引入/澄清 mid-frame pilot 与 comb-pilot stripping 术语，并更新 resource priority 文档。
  影响：通信导频、感知 pilot、CFO training field 和 payload/random QPSK 的职责边界更清楚，减少资源块配置歧义。

- 将单站/双站 fast sensing viewer 的主体逻辑重构到 `scripts/sensing_viewer/` 共享模块，并保留原启动脚本入口。
  影响：目标检测、标定、运行时参数、状态持久化和 UI 逻辑不再在两个大脚本里重复维护，后续 viewer 修改更集中。

- 恢复并增强 bistatic OS-CFAR / target detection viewer 控件，统一 clustered target marker 与 target sector 显示。
  影响：红色目标标记重新显示为聚类后的目标峰值，而不是未整理的原始检测点，AoA/Top Targets 视图也更稳定。

- 改进 Web 配置编辑器的 resource map overlay、simulation preset 支持和画布交互。
  影响：`data_resource_blocks`、`sensing_mask_blocks`、pilot 与保留字段在 UI 中更容易区分，复杂 YAML 配置的出错概率更低。

### Notes

- 第二同步符号要求 `sync_pos >= 1`，且 TX/RX 必须使用一致配置。
- CFO training field 只用于 CFO 捕获/去模糊，不是合法的 sensing-pilot 或 sensing-mask 符号；TX/RX 也必须使用一致配置。
- 维护 changelog 时应以目标分支/仓库的实际提交为准；同一功能在 `OpenISAC` 分支和公开仓库中可能对应不同 commit id。

## 2026-06-09 - V1.1

### Summary

本次更新将 OpenISAC 从硬件优先的 USRP 联调流程扩展到可在本机闭环验证的仿真/可视化工作流。核心变化是引入 ZeroMQ sensing/control 传输、ChannelSimulator 和共享内存 sim streamer，同时刷新配置、前端 viewer、benchmark 模板和公开文档。

### Changes

- 新增 `ChannelSimulator` 可执行程序和 `SimStreamer` / `ShmRing` 共享内存流式后端，支持不接 USRP 的调制端/信道/解调端闭环。
  影响：开发者可以先用本机仿真链路验证帧格式、同步、解调和感知输出，再切换到真实 USRP 硬件。

- 新增 `config/Modulator_Sim.yaml` 与 `config/Demodulator_Sim.yaml`，并在 CMake / 运行时配置中加入 `sim` backend、output/control endpoint、streamer 和 sensing 元数据参数。
  影响：仿真模式有独立示例配置，不需要改写 X310/B210 的硬件配置文件。

- 引入 `include/ZmqTransport.hpp`，将 sensing 数据与控制通道迁移到 ZeroMQ PUB / ROUTER sockets，统一后端和 Python viewer 之间的端点配置。
  影响：前端 viewer 不再依赖临时本地文件或固定进程假设，更容易做远端显示、多进程显示和控制命令扩展。

- 更新 `plot_sensing_fast.py`、`plot_bi_sensing_fast.py`、`backend_sensing_viewer.py` 和 `scripts/sensing_runtime_protocol.py`，支持新的 wire format、backend sensing metadata、面板状态持久化和多 viewer 启动方式。
  影响：单站/双站感知显示、后端聚合显示和运行时参数握手使用同一套协议约定，前端状态在重复调试时更稳定。

- 扩展 Web 配置编辑器和 schema，使 backend、ZeroMQ endpoint、sim stream、compact/resource-map 参数可以通过 UI 配置。
  影响：新增参数不需要完全手写 YAML，减少端点、资源映射和仿真配置的拼写错误。

- 刷新 benchmark 模板和工具脚本，使 BER / BLER / EVM、调制/解调 CPU 与 latency、sensing runtime benchmark 使用新的配置字段和 endpoint 约定。
  影响：性能测试脚本与主程序运行配置保持一致，后续回归测试更容易复现。

- 新增中英文 Channel Simulator 文档，并同步 README、Astro 生成数据和发布到 `docs/` 的静态文档页。
  影响：公开文档覆盖了无硬件仿真流程、配置入口和新的 viewer 启动方式。

### Notes

- 公开仓库的 `V1.1` tag 指向 commit `87a8be472211dddfc67e7e8d99abdbc682f2f16c`。
- 仿真后端用于本机开发和功能验证；真实 USRP 部署仍应使用对应 X310/B210 配置并按硬件链路验证。
- ZeroMQ viewer 依赖 `pyzmq`，请按新版 `requirements.txt` 更新 Python 环境。

## 2026-06-02 - 通信帧格式与感知响应标定

### Summary

本次更新聚焦 CPU 公共链路上的通信包格式一致性、解调端对齐，以及感知响应的幅度/相位标定能力，为后续仿真链路和前端显示协议收敛打基础。

### Changes

- 统一 CPU 调制端与解调端的 LDPC packet framing，并把公共帧打包/解包逻辑沉到 `include/OFDMCore.hpp`。
  影响：调制端和解调端不再各自维护一套容易漂移的 packet framing 规则，降低 payload 解析和 BER/BLER 统计不一致的风险。

- 新增 LDPC block interleaving 支持，并更新 X310/B210 示例配置中的相关字段。
  影响：CPU 公开链路可以使用更一致的数据块组织方式，便于做链路质量测试和后续参数 sweep。

- 改进解调端 delay alignment 和 `predictive_delay` 配置开关。
  影响：解调端可以更稳健地处理时延预测和帧边界对齐，降低同步漂移对通信/感知结果的影响。

- 新增 sensing response calibration，并将校准元数据接入后端、viewer 和 README/docs。
  影响：感知显示可以对系统响应做幅度/相位校正，更适合比较不同采集、不同通道或不同硬件设置下的结果。

- 增加 `.gitattributes` 以固定 LF checkout，并完成脚本、配置和捕获辅助文件的换行整理。
  影响：跨平台编辑和文档生成时更少出现 CRLF/LF 噪声 diff。

### Notes

- LDPC framing、block interleaving 与解调端配置应在 TX/RX 两侧保持一致。
- 感知响应标定改变的是显示/处理链路中的校准行为，不替代真实硬件链路的系统时延标定。

## 2026-05-09 - Docker 支持与 viewer 布局整理

### Summary

本次更新补充了容器化运行入口，并进一步整理 sensing viewer 的布局尺寸，让公开仓库更容易在干净环境中构建、演示和复现实验。

### Changes

- 新增 `Dockerfile`、`.dockerignore` 和 `docker-entrypoint.sh`。
  影响：可以用容器环境安装 OpenISAC 依赖并运行基础工具，降低新机器环境配置成本。

- 调整 sensing viewer 的布局 sizing。
  影响：viewer 在不同窗口尺寸下更稳定，减少控制面板、图像区域和检测结果之间的挤压。

### Notes

- Docker 支持主要用于环境复现和演示；实时 USRP 部署仍需结合宿主机 UHD、USB/网卡、CPU 亲和性和实时权限配置。

## 2026-04-19 至 2026-04-21 - 后端感知元数据、检测与显示增强

### Summary

本阶段集中增强 sensing viewer 和后端感知输出协议，加入后端 RD/CFAR/micro-Doppler 元数据、独立 viewer、显示范围控制、检测控件和 delay superresolution。

### Changes

- 新增 backend sensing metadata 处理，并加入 `backend_sensing_viewer.py`、`plot_backend_sensing.py` 和 `plot_backend_bi_sensing.py`。
  影响：后端生成的 RD 图、CFAR 检测结果和 micro-Doppler 元数据可以通过独立 viewer 检查，不必全部依赖原始 fast viewer。

- 新增 CA-CFAR target detection 逻辑和 viewer 侧检测控件。
  影响：感知显示从“只看热力图”扩展到可交互的目标检测调试，更方便评估阈值、保护单元和训练单元设置。

- 改进单站/双站 fast viewer 的显示范围、detector control、聚合 sensing pipeline 和标定显示。
  影响：不同场景下的距离/速度范围显示更可控，聚合输出与前端标定参数更一致。

- 新增 delay superresolution 显示模式。
  影响：单站感知 viewer 可以在 Delay 维度进行更细粒度的峰值观察，便于分析近距离或相邻目标。

- 停止维护旧版 `plot_sensing.py` 与 `plot_bi_sensing.py`，将维护重点转到 fast viewer。
  影响：公开前端减少重复实现，后续协议和 UI 更新集中在同一套 viewer 上。

### Notes

- 后端元数据和 fast viewer 协议需要配套更新；混用旧 viewer 与新后端时可能无法识别新增字段。
- CA-CFAR 和超分辨显示是可视化/检测辅助能力，实际参数仍应结合采样率、带宽和场景噪声重新调节。

## 2026-04-12 - CPU 链路与媒体流文档更新

### Summary

本次更新围绕 CPU 公开链路的编码/同步稳定性和辅助使用文档展开，补充 LDPC block interleaving、demodulator delay alignment、`predictive_delay` 配置和 RTP MPEG-TS 音频流命令说明。

### Changes

- CPU 调制端和解调端新增 LDPC block interleaving。
  影响：编码块组织方式更清晰，也更便于后续统一 packet framing 与链路质量统计。

- 改进 demodulator delay alignment，并新增 `predictive_delay` 开关。
  影响：解调端可以根据配置控制是否使用预测时延，降低错误对齐对解调结果的影响。

- 为 sensing viewer 增加 CA-CFAR target detection 的初始版本。
  影响：前端开始具备目标检测调试能力，为后续 detector control 和 backend metadata 链路铺路。

- 补充 RTP MPEG-TS audio streaming 命令文档。
  影响：音频流演示和外部媒体链路测试有了可复用的命令记录。

### Notes

- `predictive_delay` 和 LDPC interleaving 属于链路行为参数，修改后应同步检查调制端、解调端和 benchmark 模板。

## 2026-04-04 至 2026-04-06 - Astro 文档站点迁移与维护

### Summary

本阶段把公开文档站从静态 HTML/CSS 维护方式迁移到 Astro 源码驱动，并保留 GitHub Pages 所需的发布输出和自定义域名文件。

### Changes

- 新增 `site/` Astro 项目结构，包括页面入口、共享组件、站点数据、全局样式和 README 同步生成文件。
  影响：主页、架构页、信号处理页和中英文内容可以从结构化源码维护，而不是直接手改发布后的 HTML。

- 更新 `scripts/sync_docs_from_readme.py` 和新增 `scripts/publish_docs_site.py`，让 README 教程内容同步到 Astro，再发布到 `docs/`。
  影响：README 与网站教程内容的同步路径更明确，减少文档重复维护。

- 跟踪 Astro 构建产物所需的 `_astro` 资源、`.nojekyll`、站点图片和 `CNAME`。
  影响：GitHub Pages 可以正确发布带 hashed asset 的 Astro 静态站，并保留 `openisac.zzw123app.top` 自定义域名。

- 后续对 Astro 源码和公开 UX 文案做维护，使 `site/src/` 与已审核的首页/文档内容保持一致。
  影响：公开站点更容易继续扩展 Documentation 页面、导航和页面布局。

### Notes

- 修改 README 教程段落后，应先运行 `python3 scripts/sync_docs_from_readme.py`，再运行 `cd site && npm run build` 更新 `docs/`。
- `docs/` 是发布输出目录；常规内容变更应优先编辑 `README*` 或 `site/src/`。

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
