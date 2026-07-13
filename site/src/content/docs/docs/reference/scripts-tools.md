---
title: Scripts and Tools
description: Common repository scripts and their roles.
---

OpenISAC includes scripts for visualization, configuration, runtime control, and documentation.

## Runtime and visualization

- `scripts/plot_sensing_fast.py` displays BS-side sensing data.
- `scripts/plot_bi_sensing_fast.py` displays bistatic sensing data.
- `scripts/uplink_timing_control.py` supports timing-control workflows.
- `scripts/isolate_cpus.bash` launches backend binaries with CPU isolation.

## Configuration

- `scripts/config_web_editor.py` serves the YAML web editor.
- `scripts/config_web_editor_schema.yaml` defines editable fields.
- `scripts/config_web_editor.js` contains browser-side editor behavior.

## Documentation

- `site/` contains the Astro and Starlight documentation site.
- `scripts/publish_docs_site.py` publishes `site/dist` into root `docs/`.
