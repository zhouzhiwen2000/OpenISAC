---
title: Reference
description: Entry point for OpenISAC runtime configuration, tools, and troubleshooting.
---

Use this section to look up parameters, commands, and diagnostic procedures. If you are deploying OpenISAC for the first time, begin with [Getting Started](/docs/getting-started/hardware/). Return here when you need to change a YAML field, launch a helper tool, or diagnose a runtime problem.

## Runtime configuration

- [BS YAML Reference](./bs-yaml/) covers radio, downlink, uplink, sensing, network-output, and logging settings for the BS.
- [UE YAML Reference](./ue-yaml/) covers reception, synchronization tracking, uplink, bistatic sensing, and output settings for the UE.

Both programs read their YAML file from the current working directory and do not hot-reload it. Restart the corresponding backend after changing a runtime configuration.

## Tools and diagnostics

- [Scripts and Tools](./scripts-tools/) groups sensing viewers, timing control, CPU isolation, web configuration, and documentation commands by task.
- [Troubleshooting](./troubleshooting/) provides symptom-oriented checks for builds, YAML loading, USRP streaming, communication decoding, sensing output, and frontend connectivity.

## Recommended lookup workflow

1. Confirm whether you are editing `BS.yaml` or `UE.yaml`.
2. Search for the full YAML path, such as `uplink.enabled` or `network_output.control_port`.
3. Treat a typical value as guidance. The selected `config/*.yaml` preset and your runtime scenario remain authoritative.
4. When changing frame structure, duplex mode, frequency, or resource mapping, verify the corresponding BS and UE settings together.
