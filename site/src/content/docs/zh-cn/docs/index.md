---
title: OpenISAC 概览
description: 平台定位、能力范围、项目结构和推荐阅读路径。
---

OpenISAC 是一个基于 OFDM 的通信感知一体化平台，面向实时空口实验。它适合需要快速修改 PHY、接入 USRP 硬件、并清晰观察通信与感知处理链路的研究场景。

OpenISAC 不是 Wi-Fi、LTE 或 5G NR 协议栈，也不以商用设备互通为目标。它的价值在于把 OFDM/ISAC 想法尽快带到真实硬件实验中。

## 能力范围

- 基于 USRP 的实时 OFDM 通信与雷达式感知。
- BS 侧单站感知和 UE 侧双站感知。
- 面向分布式感知实验的 OTA 定时支持。
- YAML 驱动的运行时配置。
- 用于可视化、分析和配置的 Python 前端工具。

## 项目结构

- `src/BS.cpp` 和 `src/UE.cpp` 是 C++ 运行入口。
- `include/` 保存共享 DSP、运行时和配置类型。
- `config/` 保存硬件和仿真模式的 YAML 模板。
- `scripts/` 保存可视化、控制、配置编辑和辅助工具。
- `site/src/content/docs/` 是本文档站点的源文件。
- `docs/` 保存发布后的静态站点和独立仓库说明。

## 推荐阅读路径

先阅读[硬件准备](/zh-cn/docs/getting-started/hardware/)和[软件安装](/zh-cn/docs/getting-started/installation/)，再编译 C++ 后端并准备 `BS.yaml` / `UE.yaml`。首次 OTA 跑通后，再阅读架构和信号处理章节，理解应该在哪里修改链路。

<!-- migrated-readme-common-workflows -->
## 常见工作流

| 目标 | 后端程序 | 常用配置 | 常用前端 |
| :--- | :--- | :--- | :--- |
| 运行 BS 端 | `BS` | `config/BS_X310.yaml` 或 `config/BS_B210.yaml` | `plot_sensing_fast.py` |
| 运行 UE 端 | `UE` | `config/UE_X310.yaml` 或 `config/UE_B210.yaml` | `plot_bi_sensing_fast.py` |
| 不使用 USRP 运行 | `ChannelSimulator`、`BS`、`UE` | `config/BS_Sim.yaml` 和 `config/UE_Sim.yaml` | 见[信道仿真器](docs/CHANNEL_SIMULATOR.zh-CN.md) |
| 用浏览器调参数 | `scripts/config_web_editor.py` | 读取 `build/BS.yaml` 和 `build/UE.yaml` | 浏览器访问 `http://<host>:8765` |
