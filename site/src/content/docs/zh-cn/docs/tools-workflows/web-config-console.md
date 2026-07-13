---
title: 网页配置台
description: 在浏览器中编辑 BS/UE 配置、规划时频资源并启动运行进程。
---

网页配置台把 `build/BS.yaml` 和 `build/UE.yaml` 转换为浏览器表单，便于查看嵌套参数、切换硬件或仿真后端，以及调整通信和感知资源。保存配置后，还可以直接启动或停止 BS、UE 进程。

## 启动

在仓库根目录运行：

```bash
python3 scripts/config_web_editor.py --host 0.0.0.0 --port 8765
```

随后在浏览器打开 `http://<your-host>:8765`。仅在本机使用时，建议保留默认监听地址 `127.0.0.1`。

## 主要功能

- **配置编辑**：分别管理 BS 和 UE，以“参数 / 值”表单编辑运行时 YAML，减少手工修改长 YAML 时的缩进和字段错误。
- **资源规划**：在 `Resource Planner` 中绘制 payload 与 `sensing_pilot` 资源块，在 `Sensing Resource Map` 中选择紧凑感知输出使用的 RE。绘制区域会自动吸附到整数 RE 边界。
- **进程控制**：保存配置后，可从 `build/` 目录启动或停止 BS、UE，并可选择 CPU/CUDA 预设或自定义命令。
- **CPU 配置**：按下行、上行、感知实时循环和主线程设置 CPU 绑核，并可启用 CPU 隔离或覆盖本次隔离列表。
- **常用预设**：内置 `Guard Band Grid`，默认保留子载波 `1..489` 和 `535..N-1`，再排除同步与梳状导频资源。

## 使用说明

- 默认启动命令为 `./BS` 和 `./UE`；需要 CUDA 版本时，可在下拉框中切换。
- 编辑器直接修改 `build/` 中的运行时 YAML，因为 BS 和 UE 会从当前工作目录读取 `BS.yaml` / `UE.yaml`。
- `Resource Planner` 用来编辑 `data_resource_blocks`：它决定哪些 RE 承载 payload，哪些 RE 作为 `sensing_pilot` 保留给感知参考。
- `Sensing Resource Map` 用来编辑 `mask_blocks`：它决定 `output_mode=compact_mask` 时哪些 RE 会被送到感知输出。
- 两个规划器都可以分别应用到发射端或接收端。调试时 TX 和 RX 配置可以暂时不同，但正常收发时两端的 `data_resource_blocks` 应保持一致。
- 当 CPU 核心不足时，建议先给 `main thread affinity` 预留一个专用核心，然后优先保证 TX/RX 线程，最后再保证调制/解调线程和感知/信号处理线程；这些计算线程通常对应更大的缓冲区，对瞬时抖动更耐受。
- CPU 绑核只配置实时流水线线程和主线程。非实时的服务、输出、辅助线程有意不做绑核。
- 使用 `-1`、`[]` 或省略可选字段表示该模块不做显式绑核。
- 运行面板展示 **Isolated / Bound / Process / System** CPU 列表。默认 isolate 只覆盖最敏感线程（USRP 收发、main；BS sensing RX），不含 OFDM 调制/解调。
- 勾选 **BS+UE (isolate both sides)** 时合并两侧 YAML 的敏感核；不勾选则只 isolate 当前 tab。
- **Save + Start** / **Apply Isolation** 调用 `scripts/isolate_cpus.py`；开启隔离时进程 `AllowedCPUs` 为全部逻辑核。
- **Override isolated CPU list** 可手动指定本次 isolate 列表。
- **Reset Isolation** 将系统 slice 恢复为可用全部 CPU。
- **安全提示**：该控制台可以执行网页中填写的启动命令。请保持默认监听地址 `127.0.0.1`，或仅绑定到可信网络。
