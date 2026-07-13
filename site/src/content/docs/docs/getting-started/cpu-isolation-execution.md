---
title: CPU Isolation and Execution
description: CPU isolation workflow for stable real-time backend execution.
---

To ensure stable real-time performance, use `scripts/isolate_cpus.bash` to constrain system services (`system.slice`, `user.slice`, `init.scope`) to selected CPUs and reserve other CPUs for your workload.

All commands require root privileges (`sudo`):

```bash
cd ~/OpenISAC
chmod +x scripts/isolate_cpus.bash
sudo ./scripts/isolate_cpus.bash --help
```

**Default isolation policy**

```bash
sudo ./scripts/isolate_cpus.bash
```

- Default reserved cores for application: first 8 cores (`0-7`).
- System services are restricted to the remaining cores.
- If total CPU cores are `<= 8`, isolation cannot be applied effectively and both app/system use all cores.

**Custom application CPU set**

```bash
sudo ./scripts/isolate_cpus.bash 4          # App uses 0-3
sudo ./scripts/isolate_cpus.bash 8-15       # App uses 8-15
sudo ./scripts/isolate_cpus.bash 0,2,4,6    # App uses explicit core list
```

The selected app CPU set is saved to `/tmp/isolate_cpus_app.conf`.

**CPU binding priority when cores are limited**

- Reserve one dedicated core for the main thread first.
- Then prioritize the TX/RX real-time threads.
- Finally allocate cores to modulation/demodulation and sensing/signal-processing threads, because these compute-heavy stages typically have larger buffers and can absorb moderate scheduling jitter.

In the web CPU-binding editor, this usually means prioritizing `main thread affinity` first, then `_tx_proc` / `rx_proc` and per-channel RX loops, and only after that the modulation/demodulation and sensing-processing workers.

**Run application on reserved cores**

```bash
cd build
sudo ../scripts/isolate_cpus.bash run ./BS
```

- `run` reads saved app CPUs from `/tmp/isolate_cpus_app.conf`.
- If no saved config exists, `run` falls back to the default app CPU set.

> **Note:** Always use `sudo ../scripts/isolate_cpus.bash run ...` to launch applications after isolation is set. Direct execution or manual `taskset` may fail due to slice affinity constraints.

**Reset configuration (optional)**

```bash
sudo ./scripts/isolate_cpus.bash reset
```

This removes the isolation settings and restores system slices to all CPUs.
