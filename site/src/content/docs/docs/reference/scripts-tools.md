---
title: Scripts and Tools
description: User-facing helper scripts, launch commands, and prerequisites grouped by task.
---

This page lists tools that users launch directly. Unless stated otherwise, run commands from the repository root after installing the Python dependencies in `requirements.txt`.

## Sensing visualization

After the BS and UE backends are running and the corresponding sensing outputs are enabled, launch a viewer:

```bash
# BS multichannel monostatic sensing
python3 scripts/plot_sensing_fast.py

# UE bistatic sensing
python3 scripts/plot_bi_sensing_fast.py
```

If a window opens without data, check the sensing-output switch, IP/port settings, and backend logs before restarting the viewer.

## Uplink timing control

`uplink_timing_control.py` reads and adjusts the BS `DUTI` and UE `TADV` values through their ZMQ control ports:

```bash
python3 scripts/uplink_timing_control.py \
  --bs-host 127.0.0.1 --bs-port 9999 \
  --ue-host 127.0.0.1 --ue-port 10001
```

For remote hosts, use the actual BS and UE addresses and keep the ports aligned with `network_output.control_port`.

## CPU isolation and backend launch

`isolate_cpus.py` reads critical CPU assignments from `BS.yaml` and `UE.yaml`. Apply isolation before launching a backend through `run`:

```bash
cd build
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS
```

The default command asks whether the host runs BS, UE, or both. `run` does not restrict the whole process to isolated CPUs: YAML-pinned critical threads stay on their assigned CPUs while other threads may use system CPUs.

## Web Config Console

```bash
python3 scripts/config_web_editor.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/`. The console edits the current `build/BS.yaml` and `build/UE.yaml` and provides backend launch and CPU-isolation controls. See [Web Config Console](/docs/tools-workflows/web-config-console/) for the full workflow.

`config_web_editor_schema.yaml`, `config_web_editor.js`, and the related HTML/CSS files are internal parts of the console and are not launched separately.

## Documentation site

```bash
# Local preview
cd site
npm run dev -- --host 0.0.0.0

# Build and publish into repository-root docs/
npm run build
```

`npm run build` creates `site/dist/` and then runs `scripts/publish_docs_site.py` to update `docs/`. Do not edit generated HTML under `docs/` directly.
