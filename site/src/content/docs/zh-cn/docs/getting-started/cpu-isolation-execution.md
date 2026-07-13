---
title: CPU 隔离和执行
description: 用于稳定实时后端运行的 CPU 隔离流程。
---

为了确保稳定的实时性能，可使用 `scripts/isolate_cpus.py`（或薄封装 `scripts/isolate_cpus.bash`）将系统服务（`system.slice`、`user.slice`、`init.scope`）限制在非关键核心上，把关键 OpenISAC 线程占用的核留给实时路径。

以下设置/启动命令都需要 root 权限（`sudo`）：

```bash
cd ~/OpenISAC
chmod +x scripts/isolate_cpus.py scripts/isolate_cpus.bash
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
```

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

- `run` 将进程 `AllowedCPUs` 设为**全部**逻辑核（预留核 ∪ 系统核）。
- 关键线程通过 YAML affinity 留在预留核；非关键线程可调度到系统核。
- 若需要旧行为（整进程只在预留核），使用 `run --app-only ./BS`。

> **注意:** 完成隔离后，请始终使用 `sudo ../scripts/isolate_cpus.py run ...` 启动应用程序。直接在被限制的 user slice 中执行可能无法用上预留核。

**重置配置（可选）**

```bash
sudo ./scripts/isolate_cpus.py reset
```

该命令会移除隔离设置，并将系统 slice 恢复为可使用全部 CPU。
