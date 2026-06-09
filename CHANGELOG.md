# Changelog

本文件记录 OpenISAC 的重要更新。

当前从以下提交开始整理：

- Commit: `fe0ee68b67d6621ed22c3eaa71d1de0a7490ce3c`
- Date: `2026-04-02 22:59:43 +08:00`
- Subject: `Improve overflow/underflow recovery, add macOS support, benchmark scripts, and configurable data resource blocks`

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

- `V1.1` tag 指向 commit `87a8be472211dddfc67e7e8d99abdbc682f2f16c`。
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
