---
title: 前端与传输
description: Python 工具如何连接后端运行时。
---

OpenISAC 将实时无线链路保留在 C++ 中，将可视化、控制和分析放在 Python 工具中。

## 传输

后端到前端的数据路径和控制类工作流使用 ZeroMQ。这让前端可以替换，同时保持运行时和 UI 工具之间的稳定边界。

## 常用工具

- `scripts/plot_sensing_fast.py` 渲染 BS 侧感知输出。
- `scripts/plot_bi_sensing_fast.py` 渲染双站感知输出。
- `scripts/config_web_editor.py` 提供网页配置台。
- `scripts/uplink_timing_control.py` 支持运行时序控制工作流。

## 数据所有权

前端应消费发布的快照或控制消息，不应拥有无线时序、队列生命周期或硬实时调度决策。
