---
title: Web Config Console
description: Browser-based editor for OpenISAC YAML configuration.
---

The web config console edits runtime YAML in the browser. It makes nested YAML fields easier to inspect, reduces mistakes in long configuration files, and is useful when switching hardware targets, enabling simulation modes, or tuning sensing parameters.

## Launch

For remote-friendly configuration editing and process control, run:

```bash
python3 scripts/config_web_editor.py --host 0.0.0.0 --port 8765
```

Then open `http://<your-host>:8765` in a browser.

## What it does

- Provides separate BS / UE tabs, plus a `Resource Planner` tab for `data_resource_blocks` and a `Sensing Resource Map` tab for `mask_blocks`.
- Edits `build/BS.yaml` and `build/UE.yaml` as parameter/value forms instead of a raw YAML text area.
- Provides module-local CPU-binding fields for downlink, uplink, sensing real-time loops, and the main thread.
- Saves the current form back to YAML and starts/stops BS and UE processes from the `build/` directory.
- Includes launch options such as enabling/disabling CPU isolation and overriding the isolate CPU list.
- Includes CPU/CUDA command presets and a custom command field for each tab.
- Lets you draw payload / sensing-pilot rectangles for `data_resource_blocks`, or compact sensing rectangles for `mask_blocks`, snap the block boundaries to integer RE grid points, and apply the result independently to the transmitter or receiver YAML.
- Includes a `Guard Band Grid` preset that follows `scripts/plot_const.py`, i.e. it keeps only subcarriers `1..489` and `535..N-1` before the normal sync/pilot stripping rules are applied.

## Notes

- Default commands are `./BS` and `./UE`; switch to the CUDA preset if needed.
- The editor currently targets the runtime YAML files in `build/`, because the binaries read `BS.yaml` / `UE.yaml` from their working directory.
- `Resource Planner` edits `data_resource_blocks`: it decides which RE carry payload and which RE are reserved as `sensing_pilot`.
- `Sensing Resource Map` edits `mask_blocks`: it decides which RE are exported on the compact sensing path when `output_mode=compact_mask`.
- Both planners can be applied to either side. During experiments TX and RX may differ temporarily, but normal operation still expects matching `data_resource_blocks` on both sides.
- If CPU cores are limited, reserve a dedicated core for `main thread affinity` first, then prioritize TX/RX threads, and finally modulation/demodulation plus sensing/signal-processing threads; these compute-heavy stages typically have larger buffers and tolerate transient jitter better.
- CPU affinity is configured only for real-time pipeline threads and the main thread. Non-real-time service/output/helper threads are intentionally left unbound.
- Use `-1`, `[]`, or an omitted optional field to leave that module unbound.
- The runtime panel shows **Isolated / Bound / Process / System** CPU lists. Default isolation covers only the most sensitive threads (USRP TX/RX, main; BS sensing RX) — not OFDM mod/demod.
- **BS+UE (isolate both sides)** unions sensitive cores from both YAML files when checked; unchecked isolates only the current tab.
- **Save + Start** / **Apply Isolation** call `scripts/isolate_cpus.py`; process `AllowedCPUs` spans all logical CPUs when isolation is on.
- **Override isolated CPU list** allows a manual isolate set for this launch/apply.
- **Reset Isolation** restores system slices to all CPUs.
- Because the console can launch arbitrary commands entered in the web UI, bind it only to trusted networks or keep the default `127.0.0.1`.
