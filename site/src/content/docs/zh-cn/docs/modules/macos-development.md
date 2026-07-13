---
title: macOS 与开发说明
description: 开发机和非主运行系统的说明。
---

Linux 是 USRP 支撑的 OpenISAC 实验的主要运行环境。macOS 仍可用于文档工作、静态分析、绘图和部分开发任务。

## 推荐用途

- 编辑和构建文档站点。
- 运行不依赖 USRP 硬件的 Python 分析脚本。
- 检查 YAML 模板和生成配置。
- 查看从硬件运行中复制出的 capture。

## 限制

面向硬件的运行时行为应在实际运行 UHD 的 Linux 主机上验证。驱动可用性、调度、USB 行为和网络调优差异，使 macOS 不适合作为实时无线实验的最终验证平台。

更多 macOS 开发说明见 `docs/macos_build.md` 和 `docs/macos_build_zh.md`。
