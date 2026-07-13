---
title: 脚本与工具
description: 按用途查找 OpenISAC 常用辅助脚本、启动命令和使用条件。
---

本页只列出用户需要直接启动的工具。如无特别说明，命令都在仓库根目录执行，并需先安装 `requirements.txt` 中的 Python 依赖。

## 感知可视化

BS 和 UE 后端已经运行且开启对应感知输出后，启动查看器：

```bash
# BS 多通道单站感知
python3 scripts/plot_sensing_fast.py

# UE 双站感知
python3 scripts/plot_bi_sensing_fast.py
```

如果窗口能打开但没有数据，先检查 YAML 中的感知输出开关、IP/端口和后端日志，而不是重复启动查看器。

## 上行定时控制

`uplink_timing_control.py` 通过 BS 和 UE 的 ZMQ 控制端口读取、调整 `DUTI` 和 `TADV` 定时值：

```bash
python3 scripts/uplink_timing_control.py \
  --bs-host 127.0.0.1 --bs-port 9999 \
  --ue-host 127.0.0.1 --ue-port 10001
```

远程运行时，将 host 换成 BS/UE 的实际 IP，并确认 `network_output.control_port` 与命令一致。

## CPU 隔离与后端启动

`isolate_cpus.py` 从 `BS.yaml` / `UE.yaml` 读取关键 CPU 绑定。先应用隔离，再通过 `run` 启动后端：

```bash
cd build
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS
```

默认命令会询问本机角色是 BS、UE 还是 BS+UE。`run` 不会把整个进程限制在隔离核上；YAML 中的关键线程仍按指定 CPU 运行，其他线程可使用系统核。

## 网页配置台

```bash
python3 scripts/config_web_editor.py --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。该工具编辑当前 `build/BS.yaml` 和 `build/UE.yaml`，并提供后端启停与 CPU 隔离操作。具体使用方法见[网页配置台](/zh-cn/docs/tools-workflows/web-config-console/)。

`config_web_editor_schema.yaml`、`config_web_editor.js` 和相关 HTML/CSS 是网页配置台的内部组成文件，不需要单独执行。

## 文档站

```bash
# 本地预览
cd site
npm run dev -- --host 0.0.0.0

# 构建并发布到仓库根目录 docs/
npm run build
```

`npm run build` 会先生成 `site/dist/`，再调用 `scripts/publish_docs_site.py` 更新 `docs/`。不要直接修改 `docs/` 下的生成 HTML。
