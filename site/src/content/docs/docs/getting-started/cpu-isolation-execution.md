---
title: CPU Isolation and Execution
description: CPU isolation workflow for stable real-time backend execution.
---

To ensure stable real-time performance, use `scripts/isolate_cpus.py` to constrain system services (`system.slice`, `user.slice`, `init.scope`) away from critical OpenISAC cores.

All setup/run commands require root privileges (`sudo`):

```bash
cd ~/OpenISAC
chmod +x scripts/isolate_cpus.py
sudo ./scripts/isolate_cpus.py --help
```

**Default isolation policy (YAML-driven)**

```bash
cd build
# Ensure BS.yaml / UE.yaml exist (copy from config/ as needed)
sudo ../scripts/isolate_cpus.py
```

The script first asks which role this machine has:

1. **BS only** — reads `BS.yaml`
2. **UE only** — reads `UE.yaml`
3. **BS + UE** — reads both (same-host dual stack)

If `/sys/devices/system/cpu/isolated` contains any logical CPU, an interactive second menu asks how to handle a detected 10 GbE-or-faster NIC:

1. **Disable / skip NIC tuning** — do not enable NIC isolation; if a previous saved run enabled it, restore that NIC state first.
2. **IRQ isolation only** — preserve the NIC's existing combined queues and pin their IRQs to one kernel-isolated physical core.
3. **IRQ + RX/TX split** — shown only when at least two eligible isolated physical cores and a YAML-pinned USRP TX CPU are available; use two combined queues, route RX to queue 0 and TX threads to queue 1, then pin the two queue IRQs to separate physical cores.

NIC IRQ CPUs and their SMT siblings are removed from both system-slice CPUs and backend-process `AllowedCPUs`, so application threads cannot contend for the same physical cores as the IRQ handlers. Candidates that share a physical core with a YAML-reserved application thread are rejected. The split mode reads BS TX from `downlink_cpu_cores[0]` and UE TX from `uplink_cpu_cores[2]`; with `--role both`, both TX cores map to TX queue 1. Active `irqbalance` is stopped while this configuration is in use.

It then reserves only the **most scheduling-sensitive** cores from YAML:

- USRP sample TX/RX threads (e.g. BS TX, UE `rx_proc`, uplink TX/RX ingest)
- `main_cpu_core`
- BS monostatic sensing `rx_cpu_core` (sample ingest only)

OFDM modulation/demodulation, LDPC, UDP, sensing processing, and workers are **not** reserved by default so they can share noisier system cores.

Non-interactive / scripted:

```bash
sudo ../scripts/isolate_cpus.py --role bs
sudo ../scripts/isolate_cpus.py --role ue
sudo ../scripts/isolate_cpus.py --role both
sudo ../scripts/isolate_cpus.py show-plan --role bs   # dry-run

# Non-interactive NIC policy
sudo ../scripts/isolate_cpus.py --role ue --nic-irq split
sudo ../scripts/isolate_cpus.py --role bs --nic-irq isolate
sudo ../scripts/isolate_cpus.py --role both --nic-irq off
```

`--nic-irq ask` is the default and only opens the NIC menu on a TTY. Scripts and the web console therefore skip NIC IRQ changes unless they explicitly pass `--nic-irq isolate|split`. Use `--nic-ifaces ens1f1` (comma-separated for multiple interfaces) to override automatic active high-speed NIC detection.

**Custom application CPU set (manual override)**

```bash
sudo ./scripts/isolate_cpus.py 4          # App uses 0-3
sudo ./scripts/isolate_cpus.py 8-15       # App uses 8-15
sudo ./scripts/isolate_cpus.py 0,2,4,6    # App uses explicit core list
```

State is saved to `/tmp/isolate_cpus_app.conf` (reserved cores) and `/tmp/isolate_cpus_state.json` (full plan).

**CPU binding priority when cores are limited**

- Reserve one dedicated core for the main thread first.
- Then prioritize the TX/RX real-time threads.
- Then OFDM modulation/demodulation threads.
- Leave LDPC / UDP / sensing-processing on system cores when possible.

**Run application (process may use all CPUs)**

```bash
cd build
sudo ../scripts/isolate_cpus.py run ./BS
sudo ../scripts/isolate_cpus.py run ./UE
```

- `run` normally sets process `AllowedCPUs` to all logical CPUs; when NIC IRQ isolation is enabled, it excludes the selected IRQ CPUs and their SMT siblings.
- Critical threads stay on reserved cores via YAML affinity; non-critical threads may schedule on system cores.
- Use `run --app-only ./BS` only if you want the old whole-process reserved-only behavior.

> **Note:** Always launch via `sudo ../scripts/isolate_cpus.py run ...` after isolation is set. Direct execution from a restricted user slice may fail to use reserved cores.

**Reset configuration (optional)**

```bash
sudo ./scripts/isolate_cpus.py reset
```

This restores system slices to all CPUs. If the saved state includes NIC IRQ isolation, it also restores the prior combined-queue count, resets RSS/XPS steering, releases IRQ affinity, and restarts `irqbalance` when this script had stopped it.

`set_performance.bash` still configures socket buffers, NIC ring sizes, MTU, CPU governor, and real-time throttling. It no longer changes combined queues or NIC IRQ affinity; those settings belong exclusively to this script.
