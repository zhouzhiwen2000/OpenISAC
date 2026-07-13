---
title: CPU Isolation and Execution
description: CPU isolation workflow for stable real-time backend execution.
---

To ensure stable real-time performance, use `scripts/isolate_cpus.py` (or the thin wrapper `scripts/isolate_cpus.bash`) to constrain system services (`system.slice`, `user.slice`, `init.scope`) away from critical OpenISAC cores.

All setup/run commands require root privileges (`sudo`):

```bash
cd ~/OpenISAC
chmod +x scripts/isolate_cpus.py scripts/isolate_cpus.bash
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
```

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

- `run` sets process `AllowedCPUs` to **all** logical CPUs (reserved ∪ system).
- Critical threads stay on reserved cores via YAML affinity; non-critical threads may schedule on system cores.
- Use `run --app-only ./BS` only if you want the old whole-process reserved-only behavior.

> **Note:** Always launch via `sudo ../scripts/isolate_cpus.py run ...` after isolation is set. Direct execution from a restricted user slice may fail to use reserved cores.

**Reset configuration (optional)**

```bash
sudo ./scripts/isolate_cpus.py reset
```

This removes the isolation settings and restores system slices to all CPUs.
