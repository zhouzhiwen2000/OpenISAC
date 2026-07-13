---
title: OpenISAC Overview
description: Platform positioning, capabilities, project map, and recommended reading path.
---

OpenISAC is an OFDM-based integrated sensing and communication platform for real-time over-the-air experiments. It is designed for researchers who want a compact PHY-level system that can be modified quickly, run on USRP hardware, and expose the sensing and communication paths clearly.

OpenISAC is not a Wi-Fi, LTE, or 5G NR stack. It does not target interoperability with commercial devices. Its value is the short path from an OFDM or ISAC idea to a hardware experiment.

## Capabilities

- Real-time OFDM communication and radar-style sensing on USRP devices.
- BS-side monostatic sensing and UE-side bistatic sensing.
- OTA timing support for distributed sensing experiments.
- YAML-driven runtime configuration.
- Python frontends for visualization, analysis, and configuration.

## Project map

- `src/BS.cpp` and `src/UE.cpp` are the C++ runtime entry points.
- `include/` contains shared DSP, runtime, and configuration types.
- `config/` contains YAML templates for supported hardware and simulation modes.
- `scripts/` contains visualization, control, config editing, and helper tools.
- `site/src/content/docs/` is the source of this documentation site.
- `docs/` contains published static site output and standalone repository notes.

## Recommended reading path

Start with [hardware setup](/docs/getting-started/hardware/) and [installation](/docs/getting-started/installation/), then build the C++ runtime and prepare `BS.yaml` / `UE.yaml`. After the first OTA run, read the architecture and signal-processing sections to understand where to modify the pipeline.

<!-- migrated-readme-common-workflows -->
## Common Workflows

| Goal | Backend program | Typical config | Typical frontend |
| :--- | :--- | :--- | :--- |
| Run the BS side | `BS` | `config/BS_X310.yaml` or `config/BS_B210.yaml` | `plot_sensing_fast.py` |
| Run the UE side | `UE` | `config/UE_X310.yaml` or `config/UE_B210.yaml` | `plot_bi_sensing_fast.py` |
| Run without USRPs | `ChannelSimulator`, `BS`, `UE` | `config/BS_Sim.yaml` and `config/UE_Sim.yaml` | See [Channel Simulator](./tools-workflows/channel-simulator/) |
| Tune parameters from a browser | `scripts/config_web_editor.py` | Reads `build/BS.yaml` and `build/UE.yaml` | Browser at `http://<host>:8765` |
