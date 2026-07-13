---
title: 脚本与工具
description: 常见仓库脚本及其作用。
---

OpenISAC 包含用于可视化、配置、运行时控制和文档的脚本。

## 运行和可视化

- `scripts/plot_sensing_fast.py` 显示 BS 侧感知数据。
- `scripts/plot_bi_sensing_fast.py` 显示双站感知数据。
- `scripts/uplink_timing_control.py` 支持时序控制工作流。
- `scripts/isolate_cpus.py`（封装：`isolate_cpus.bash`）按 YAML 关键核做 CPU 隔离，并用 `run` 启动后端（进程可用全部 CPU）。

## 配置

- `scripts/config_web_editor.py` 提供 YAML 网页编辑器。
- `scripts/config_web_editor_schema.yaml` 定义可编辑字段。
- `scripts/config_web_editor.js` 包含浏览器侧编辑器行为。

## 文档

- `site/` 包含 Astro 和 Starlight 文档站点。
- `scripts/publish_docs_site.py` 将 `site/dist` 发布到根目录 `docs/`。
