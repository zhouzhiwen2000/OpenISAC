---
title: 网页配置台
description: 用于 OpenISAC YAML 配置的浏览器编辑器。
---

网页配置台用于在浏览器中编辑运行时 YAML。它让嵌套 YAML 字段更容易检查，并降低长配置文件中的错误；切换硬件目标、启用仿真模式或调感知参数时尤其有用。

## 启动

如果希望通过浏览器远程修改配置并控制进程，可运行：

```bash
python3 scripts/config_web_editor.py --host 0.0.0.0 --port 8765
```

随后在浏览器打开 `http://<your-host>:8765`。

## 功能

- BS 和 UE 使用不同 tab 分开管理，并额外提供 `Resource Planner` 和 `Sensing Resource Map` 两个规划 tab，分别对应 `data_resource_blocks` 与 `mask_blocks`。
- 以“参数 / 值”表单方式编辑 `build/BS.yaml` 和 `build/UE.yaml`，而不是原始 YAML 文本框。
- 提供按模块放置的 CPU 绑核字段，覆盖下行、上行、感知实时 loop 和主线程模块。
- 保存当前表单后，可在 `build/` 目录中启动/停止 BS 与 UE 进程。
- 提供启动相关选项，例如是否启用 CPU 隔离、以及是否覆盖默认的 isolate CPU 列表。
- 每个 tab 都提供 CPU/CUDA 预设命令，也支持自定义启动命令。
- 可以在较大的时频资源网格画布上直接绘制 `data_resource_blocks` 的 payload / sensing-pilot 矩形块，或绘制 `mask_blocks` 的紧凑感知矩形块；绘制结果会吸附到整数 RE 格点边界，并可分别应用到发射端或接收端 YAML。
- 内置 `Guard Band Grid` 预设，规则与 `scripts/plot_const.py` 一致：默认仅保留 `1..489` 和 `535..N-1` 这两段子载波，然后再继续套用同步 / 梳状导频的剔除规则。

## 说明

- 默认命令分别是 `./BS` 和 `./UE`；如果需要 CUDA 版本，可在下拉框里切换。
- 编辑器当前直接面向 `build/` 目录中的运行时 YAML，因为二进制程序会从各自工作目录读取 `BS.yaml` / `UE.yaml`。
- `Resource Planner` 用来编辑 `data_resource_blocks`：它决定哪些 RE 承载 payload，哪些 RE 作为 `sensing_pilot` 保留给感知参考。
- `Sensing Resource Map` 用来编辑 `mask_blocks`：它决定 `output_mode=compact_mask` 时哪些 RE 会被送到感知输出。
- 两个 planner 都可以分别应用到发射端或接收端。实验时 TX 和 RX 可以暂时不同，但正常收发时 `data_resource_blocks` 仍应保持一致。
- 当 CPU 核心不足时，建议先给 `main thread affinity` 预留一个专用核心，然后优先保证 TX/RX 线程，最后再保证调制/解调线程和感知/信号处理线程；这些计算线程通常对应更大的缓冲区，对瞬时抖动更耐受。
- CPU 绑核只配置实时流水线线程和主线程。非实时的服务、输出、辅助线程有意不做绑核。
- 使用 `-1`、`[]` 或省略可选字段表示该模块不做显式绑核。
- 若开启 `Enable runtime CPU isolation`，控制台会根据所有非负的实时 CPU 字段共同计算默认 isolate CPU 列表，并在启动前调用 `scripts/isolate_cpus.bash`。
- 若再开启 `Override CPU isolation list`，右侧文本框会先用默认 isolate 列表初始化，然后允许按本次启动需要手工修改。
- 若关闭 `Enable runtime CPU isolation`，控制台仍会通过特权运行路径启动选中的命令，但不会调用 `scripts/isolate_cpus.bash`。
- 运行面板还提供可选的 sudo 密码输入框，以及 `Reset CPU isolation` 操作。
- 该控制台可以执行网页中输入的启动命令，因此只应绑定到可信网络，或者保持默认 `127.0.0.1`。
