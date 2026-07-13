<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/logo.svg">
    <img src="docs/images/logo.svg" alt="OpenISAC Logo" width="400">
  </picture>
</p>

# OpenISAC

[中文版本](README_zh.md) | [Changelog](CHANGELOG.md)

OpenISAC is a real-time OFDM platform for integrated sensing and communication (ISAC), built for academic experiments and rapid PHY iteration.

It is designed for the common gap between simulation code and full standard stacks: simple enough to modify quickly, but complete enough to run over the air with USRP hardware.

If your goal is "idea -> OTA experiment" with a minimal and readable codebase, this repository is a good fit. If you need Wi-Fi/5G NR interoperability or a production-grade protocol stack, it is not.

## Highlights

- Real-time OFDM communication plus monostatic and bistatic sensing
- Over-the-air synchronization support for bistatic experiments
- C++ backend for the radio path and Python frontend tools for visualization
- YAML-based runtime configuration with sample presets for X310/B210, adaptable to other UHD-supported USRPs
- Included utilities for CPU isolation, plotting, web-based config editing, and process control

## At a Glance

| Component | Main entry points | Purpose |
| :--- | :--- | :--- |
| BS backend | `BS`, `config/BS_*.yaml` | Transmit OFDM frames, ingest UDP payloads, and output monostatic sensing data |
| UE backend | `UE`, `config/UE_*.yaml` | Receive and decode frames, output payload data, and run bistatic sensing |
| Frontend tools | `scripts/plot_*.py`, `scripts/config_web_editor.py` | Visualize sensing/channel results and edit runtime configs |

## Quick Navigation

- Documentation: [English manual](https://openisac.zzw123app.top/docs/) and [Chinese manual](https://openisac.zzw123app.top/zh-cn/docs/)
- Setup and installation: [Hardware](https://openisac.zzw123app.top/docs/getting-started/hardware/), [Installation](https://openisac.zzw123app.top/docs/getting-started/installation/), [Build](https://openisac.zzw123app.top/docs/getting-started/build/)
- First end-to-end run: [First OTA Run](https://openisac.zzw123app.top/docs/getting-started/first-ota-run/)
- Runtime configuration: [BS YAML Reference](https://openisac.zzw123app.top/docs/reference/bs-yaml/) and [UE YAML Reference](https://openisac.zzw123app.top/docs/reference/ue-yaml/)
- USRP-free simulation: [Channel Simulator](docs/CHANNEL_SIMULATOR.md) and [Starlight notes](https://openisac.zzw123app.top/docs/tools-workflows/channel-simulator/)
- Web control UI: [Web Config Console](https://openisac.zzw123app.top/docs/tools-workflows/web-config-console/)
- Recent updates: [Changelog](CHANGELOG.md)

## Repository Layout

| Path | Description |
| :--- | :--- |
| `src/`, `include/` | Core C++ PHY, sensing, threading, and runtime logic |
| `config/` | Sample YAML presets for different roles, including X310/B210 examples that can be adapted to other USRPs |
| `scripts/` | Python frontends, web config console, and Linux performance helpers |
| `capture/` | Offline plotting helpers for saved sensing results |
| `docs/` | Static project site and architecture/signal-processing pages |

## What it is - and what it is not

### What OpenISAC is

- A minimal OFDM-based PHY for joint communication and sensing research
- Designed for prototyping, academic experiments, and rapid algorithm validation
- Focused on readability and modification speed rather than full-stack completeness

### What OpenISAC is not

- A standard-compliant implementation; it does not aim to match Wi-Fi or 5G NR
- A replacement for full-stack systems such as openwifi or OpenAirInterface
- A production-ready communications stack

### When to use it

- Prototyping new OFDM/ISAC algorithms
- Rapidly validating PHY, synchronization, or sensing ideas over the air
- Research setups where interoperability is not required

### When not to use it

- Building a Wi-Fi/NR-compatible system
- Requiring full MAC/stack behavior, interoperability, or certification-oriented behavior

## Citation

If you find this repository useful, please cite our paper:

> Z. Zhou, C. Zhang, X. Xu, and Y. Zeng, "OpenISAC: An Open-Source Real-Time Experimentation Platform for OFDM-ISAC with Over-the-Air Synchronization," *arXiv preprint* arXiv:2601.03535, Jan. 2026.
>
> [[arXiv](https://arxiv.org/pdf/2601.03535)]

## Authors

- Zhiwen Zhou (zhiwen_zhou@seu.edu.cn)
- Chaoyue Zhang (chaoyue_zhang@seu.edu.cn)
- Xiaoli Xu (Member, IEEE) (xiaolixu@seu.edu.cn)
- Yong Zeng (Fellow, IEEE) (yong_zeng@seu.edu.cn)

## Affiliation

<img src="docs/images/SEUlogo.png" height="80" alt="SEU Logo" style="border:none; box-shadow:none;"> &nbsp;&nbsp; <img src="docs/images/PML.png" height="80" alt="PML Logo" style="border:none; box-shadow:none;">

**Yong Zeng Group at the National Mobile Communications Research Laboratory, Southeast University and the Purple Mountain Laboratories**

## Community

- [Join our QQ Group](https://qm.qq.com/q/NIQRNGb0kY)
- [Bilibili Channel (Yong Zeng Group)](https://space.bilibili.com/627920129)
- WeChat Official Account:

  <img src="docs/images/WeChat.jpg" width="150" alt="WeChat QR Code">
