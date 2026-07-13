<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/logo.svg">
    <img src="docs/images/logo.svg" alt="OpenISAC Logo" width="400">
  </picture>
</p>

# OpenISAC

[English Version](README.md) | [更新日志](CHANGELOG.md)

OpenISAC 是一个面向实时实验的 OFDM 通信感知一体化（ISAC）平台，重点服务于学术研究和 PHY 层快速迭代。

它试图填补“纯仿真代码”和“完整标准协议栈”之间的空档：既尽量保持代码简洁、容易修改，又能基于 USRP 直接开展空口实验。

如果你的目标是用一个足够轻量、易读、可快速改造的系统，把算法思路尽快跑到 OTA 实验里，这个仓库适合你；如果你需要 Wi-Fi/5G NR 互操作性或生产级协议栈，它并不适合。

## 项目亮点

- 支持实时 OFDM 通信，以及单站和双站感知
- 支持双站实验所需的空口同步机制
- 后端采用 C++ 实时链路，前端提供 Python 可视化与工具脚本
- 基于 YAML 的运行时配置，仓库内提供 X310/B210 样例，也可扩展到其他 UHD 支持的 USRP
- 自带 CPU 隔离、绘图分析、网页配置与进程控制工具

## 一眼看懂

| 组件 | 主要入口 | 作用 |
| :--- | :--- | :--- |
| BS 后端 | `BS`、`config/BS_*.yaml` | 发送 OFDM 帧、接收业务 UDP，并输出单站感知结果 |
| UE 后端 | `UE`、`config/UE_*.yaml` | 接收解调 OFDM 帧、输出业务数据，并运行双站感知 |
| 前端工具 | `scripts/plot_*.py`、`scripts/config_web_editor.py` | 显示感知/信道结果，并编辑运行时配置 |

## 快速导航

- 文档站点：[英文手册](https://openisac.zzw123app.top/docs/) 和 [中文手册](https://openisac.zzw123app.top/zh-cn/docs/)
- 环境准备与安装：[硬件准备](https://openisac.zzw123app.top/zh-cn/docs/getting-started/hardware/)、[软件安装](https://openisac.zzw123app.top/zh-cn/docs/getting-started/installation/)、[编译](https://openisac.zzw123app.top/zh-cn/docs/getting-started/build/)
- 先跑起来：[首次 OTA 运行](https://openisac.zzw123app.top/zh-cn/docs/getting-started/first-ota-run/)
- 运行参数说明：[BS YAML 参考](https://openisac.zzw123app.top/zh-cn/docs/reference/bs-yaml/) 和 [UE YAML 参考](https://openisac.zzw123app.top/zh-cn/docs/reference/ue-yaml/)
- 无 USRP 仿真：[信道仿真器](docs/CHANNEL_SIMULATOR.zh-CN.md) 和 [Starlight 说明](https://openisac.zzw123app.top/zh-cn/docs/tools-workflows/channel-simulator/)
- 网页控制台：[Web Config Console](https://openisac.zzw123app.top/zh-cn/docs/tools-workflows/web-config-console/)
- 最近更新：[更新日志](CHANGELOG.md)

## 仓库结构

| 路径 | 说明 |
| :--- | :--- |
| `src/`、`include/` | 核心 C++ PHY、感知、线程与运行时逻辑 |
| `config/` | 不同角色的 YAML 样例配置，内含 X310/B210 示例，也可作为其他 USRP 的起点 |
| `scripts/` | Python 前端、网页配置控制台、Linux 性能调优脚本 |
| `capture/` | 离线感知结果绘图工具 |
| `docs/` | 项目静态站点，以及架构/信号处理说明页 |

## 它是什么，又不是什么？

### OpenISAC 是什么？

- 一个面向通信感知一体化研究的极简 OFDM PHY
- 适合原型验证、学术实验与快速算法迭代
- 更强调代码可读性、易改性和实验效率，而不是全栈完备性

### OpenISAC 不是什么？

- 不是标准兼容实现，不以 Wi-Fi 或 5G NR 兼容为目标
- 不是 openwifi、OpenAirInterface 这类全栈系统的替代品
- 不是生产级通信协议栈

### 何时使用它

- 快速验证新的 OFDM/ISAC 算法
- 将同步、感知或 PHY 思路尽快跑到空口实验
- 不要求互操作性的科研环境

### 何时不使用它

- 需要构建兼容 Wi-Fi/NR 的系统
- 需要完整 MAC/协议栈、互操作性或面向认证的行为

## 引用

如果您觉得这个仓库有用，请引用我们的论文：

> Z. Zhou, C. Zhang, X. Xu, and Y. Zeng, "OpenISAC: An Open-Source Real-Time Experimentation Platform for OFDM-ISAC with Over-the-Air Synchronization," *arXiv preprint* arXiv:2601.03535, Jan. 2026.
>
> [[arXiv](https://arxiv.org/pdf/2601.03535)]

## 作者

- 周智文 (zhiwen_zhou@seu.edu.cn)
- 张超越 (chaoyue_zhang@seu.edu.cn)
- 徐晓莉 (Member, IEEE) (xiaolixu@seu.edu.cn)
- 曾勇 (Fellow, IEEE) (yong_zeng@seu.edu.cn)

## 所属机构

<img src="docs/images/SEUlogo.png" height="80" alt="SEU Logo" style="border:none; box-shadow:none;"> &nbsp;&nbsp; <img src="docs/images/PML.png" height="80" alt="PML Logo" style="border:none; box-shadow:none;">

**东南大学移动通信国家重点实验室 & 紫金山实验室 曾勇课题组**

## 社区

- [加入我们的 QQ 群](https://qm.qq.com/q/NIQRNGb0kY)
- [Bilibili 频道 (曾勇课题组)](https://space.bilibili.com/627920129)
- 微信公众号:

  <img src="docs/images/WeChat.jpg" width="150" alt="WeChat QR Code">
