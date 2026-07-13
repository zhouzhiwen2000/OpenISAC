---
title: System Architecture
description: Hardware and software roles in the OpenISAC system.
---

OpenISAC is built around a BS node and a UE node. Each node combines a host PC, a USRP, runtime YAML configuration, and a C++ processing pipeline.

![OpenISAC system architecture](/images/SysArch.png)

## BS node

The BS generates the downlink OFDM waveform, streams it to the USRP, receives sensing echoes, and computes monostatic sensing products. It also handles payload ingress and frontend data publication.

## UE node

The UE receives the downlink waveform, estimates synchronization and channel state, decodes communication payloads, and can publish bistatic sensing data. Runtime timing information is also used by OTA/eRTM timing workflows.

## Frontend tools

Python tools receive sensing or status streams over ZeroMQ, render plots, and provide control/configuration workflows. The frontend is intentionally separated from the hard real-time radio path.
