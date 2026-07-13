---
title: CPU 隔离和执行
description: 用于稳定实时后端运行的 CPU 隔离流程。
---

为了确保稳定的实时性能，可使用 `scripts/isolate_cpus.bash` 将系统服务 (`system.slice`、`user.slice`、`init.scope`) 限制到指定 CPU，并将其余 CPU 保留给应用程序。

以下命令都需要 root 权限（`sudo`）：

```bash
cd ~/OpenISAC
chmod +x scripts/isolate_cpus.bash
sudo ./scripts/isolate_cpus.bash --help
```

**默认隔离策略**

```bash
sudo ./scripts/isolate_cpus.bash
```

- 默认给应用预留前 8 个核心（`0-7`）。
- 系统服务会被限制到其余核心。
- 如果总核心数 `<= 8`，则无法有效隔离，应用与系统都会使用全部核心。

**自定义应用 CPU 集合**

```bash
sudo ./scripts/isolate_cpus.bash 4          # 应用使用 0-3
sudo ./scripts/isolate_cpus.bash 8-15       # 应用使用 8-15
sudo ./scripts/isolate_cpus.bash 0,2,4,6    # 应用使用显式核心列表
```

脚本会将应用核心配置保存到 `/tmp/isolate_cpus_app.conf`。

**CPU 绑核优先级（当核心数量紧张时）**

- 首先给主线程预留一个专用核心。
- 其次优先保证 TX/RX 实时线程。
- 最后再分配调制/解调线程和感知/信号处理线程，因为这些计算线程通常对应更大的缓冲区，可以吸收一定的调度抖动。

在网页 CPU 绑核编辑器里，通常意味着优先保证 `main thread affinity`，其次是 `_tx_proc` / `rx_proc` 和各通道 RX loop，最后再考虑调制、解调和感知处理相关线程。

**在预留核心上运行应用**

```bash
cd build
sudo ../scripts/isolate_cpus.bash run ./BS
```

- `run` 会优先读取 `/tmp/isolate_cpus_app.conf` 中保存的应用核心配置。
- 若没有保存配置，`run` 会回退到默认应用核心集合。

> **注意:** 完成隔离后，请始终使用 `sudo ../scripts/isolate_cpus.bash run ...` 启动应用程序。直接执行或手动使用 `taskset` 可能因 slice 亲和性限制而失败。

**重置配置（可选）**

```bash
sudo ./scripts/isolate_cpus.bash reset
```

该命令会移除隔离设置，并将系统 slice 恢复为可使用全部 CPU。
