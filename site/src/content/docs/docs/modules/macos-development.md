---
title: macOS and Development Notes
description: Notes for development machines and non-primary operating systems.
---

Linux is the primary runtime environment for USRP-backed OpenISAC experiments. macOS can still be useful for documentation work, static analysis, plotting, and some development tasks.

## Recommended use

- Edit and build the documentation site.
- Run Python analysis scripts that do not require USRP hardware.
- Review YAML templates and generated configs.
- Inspect captures copied from hardware runs.

## Limitations

Hardware-facing runtime behavior should be validated on the Linux host that will run UHD. Differences in driver availability, scheduling, USB handling, and network tuning make macOS unsuitable as the final validation platform for real-time radio experiments.

See `docs/macos_build.md` and `docs/macos_build_zh.md` for repository notes specific to macOS development.
