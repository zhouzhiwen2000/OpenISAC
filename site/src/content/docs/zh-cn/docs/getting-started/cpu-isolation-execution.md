---
title: CPU 隔离和执行
description: 用于稳定实时后端运行的 CPU 隔离流程。
---

为了确保稳定的实时性能，可使用 `scripts/isolate_cpus.py` 将系统服务（`system.slice`、`user.slice`、`init.scope`）限制在非关键核心上，把关键 OpenISAC 线程占用的核留给实时路径。

以下设置/启动命令都需要 root 权限（`sudo`）：

```bash
cd ~/OpenISAC
chmod +x scripts/isolate_cpus.py
sudo ./scripts/isolate_cpus.py --help
```

**默认隔离策略（基于 YAML）**

```bash
cd build
# 确保目录下有 BS.yaml / UE.yaml（可从 config/ 复制）
sudo ../scripts/isolate_cpus.py
```

脚本会先询问本机角色：

1. **仅 BS** — 读取 `BS.yaml`
2. **仅 UE** — 读取 `UE.yaml`
3. **BS + UE** — 同机双端，读取两者

如果 `/sys/devices/system/cpu/isolated` 中发现任意逻辑 CPU，交互运行时就会显示第二级菜单，询问如何处理检测到的 10 GbE 或更高速网卡：

1. **禁用 / 跳过网卡调优** — 不启用网卡隔离；若上次保存的运行已启用，则先恢复此前的网卡状态。
2. **仅隔离 IRQ** — 保留网卡现有 combined 队列，把这些队列的 IRQ 固定到一个内核隔离的物理核心。
3. **IRQ + RX/TX 分流** — 仅在至少有两个可用的隔离物理核心且 YAML 配置了 USRP TX 核时显示；使用两个 combined 队列，RX 全部进入 queue 0，TX 线程进入 queue 1，再把两个队列 IRQ 固定到不同物理核心。

网卡 IRQ CPU 及其 SMT 同胞会同时从系统 slice 和后端进程的 `AllowedCPUs` 中排除，避免应用线程与 IRQ handler 争用同一物理核心；与 YAML 实时线程同属一个物理核心的候选也不会被选中。分流模式从 `downlink_cpu_cores[0]` 读取 BS TX 核，从 `uplink_cpu_cores[2]` 读取 UE TX 核；使用 `--role both` 时会把两个 TX 核都映射到 TX queue 1。若 `irqbalance` 正在运行，脚本会在应用该配置时停止它。

然后只预留 YAML 中**对调度最敏感**的核：

- USRP 样本收发线程（如 BS TX、UE `rx_proc`、上行 TX/RX ingest）
- `main_cpu_core`
- BS 单站感知 `rx_cpu_core`（仅样本采集）

OFDM 调制/解调、LDPC、UDP、感知处理、worker 等**默认不预留**，可与系统核共享。

非交互 / 脚本用法：

```bash
sudo ../scripts/isolate_cpus.py --role bs
sudo ../scripts/isolate_cpus.py --role ue
sudo ../scripts/isolate_cpus.py --role both
sudo ../scripts/isolate_cpus.py show-plan --role bs   # 只打印计划

# 非交互网卡策略
sudo ../scripts/isolate_cpus.py --role ue --nic-irq split
sudo ../scripts/isolate_cpus.py --role bs --nic-irq isolate
sudo ../scripts/isolate_cpus.py --role both --nic-irq off
```

默认值 `--nic-irq ask` 只会在 TTY 中打开网卡菜单，因此脚本和网页配置台不会在未明确指定时修改网卡 IRQ。可用 `--nic-ifaces ens1f1` 限定网卡；多块网卡用逗号分隔。

**自定义应用 CPU 集合（手动覆盖）**

```bash
sudo ./scripts/isolate_cpus.py 4          # 应用使用 0-3
sudo ./scripts/isolate_cpus.py 8-15       # 应用使用 8-15
sudo ./scripts/isolate_cpus.py 0,2,4,6    # 应用使用显式核心列表
```

状态会写入 `/tmp/isolate_cpus_app.conf`（预留核）和 `/tmp/isolate_cpus_state.json`（完整计划）。

**CPU 绑核优先级（当核心数量紧张时）**

- 首先给主线程预留一个专用核心。
- 其次优先保证 TX/RX 实时线程。
- 然后是 OFDM 调制/解调线程。
- LDPC / UDP / 感知处理尽量放在系统核。

**运行应用（进程可使用全部 CPU）**

```bash
cd build
sudo ../scripts/isolate_cpus.py run ./BS
sudo ../scripts/isolate_cpus.py run ./UE
```

- `run` 通常将进程 `AllowedCPUs` 设为全部逻辑核；启用网卡 IRQ 隔离后，会排除所选 IRQ CPU 及其 SMT 同胞。
- 关键线程通过 YAML affinity 留在预留核；非关键线程可调度到系统核。
- 若需要旧行为（整进程只在预留核），使用 `run --app-only ./BS`。

> **注意:** 完成隔离后，请始终使用 `sudo ../scripts/isolate_cpus.py run ...` 启动应用程序。直接在被限制的 user slice 中执行可能无法用上预留核。

**重置配置（可选）**

```bash
sudo ./scripts/isolate_cpus.py reset
```

该命令会将系统 slice 恢复为可使用全部 CPU。若保存状态中包含网卡 IRQ 隔离，还会恢复原 combined 队列数、重置 RSS/XPS steering、释放 IRQ affinity，并在脚本此前停止过 `irqbalance` 时重新启动它。

`set_performance.bash` 仍负责 socket buffer、网卡 ring、MTU、CPU governor 和实时 throttling，但不再修改 combined 队列或网卡 IRQ affinity；这些设置现在统一由本脚本负责。
