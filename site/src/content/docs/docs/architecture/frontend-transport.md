---
title: Frontend and Transport
description: How Python tools connect to the backend runtime.
---

OpenISAC keeps real-time radio work in C++ and moves visualization, control, and analysis into Python tools.

## Transport

ZeroMQ is used for backend-to-frontend data paths and control-style workflows. This keeps the frontend replaceable while preserving a stable boundary between the runtime and UI tools.

## Common tools

- `scripts/plot_sensing_fast.py` renders real-time sensing output.
- `scripts/plot_bi_sensing_fast.py` renders bistatic sensing output.
- `scripts/config_web_editor.py` serves the web configuration console.
- `scripts/uplink_timing_control.py` supports runtime timing workflows.

## Data ownership

The frontend should consume published snapshots or control messages. It should not own radio timing, queue lifecycle, or hard real-time scheduling decisions.
