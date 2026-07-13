---
title: Frontend and Transport
description: Data and control paths between the real-time C++ backend and Python interaction layer.
---

OpenISAC uses a layered C++ and Python architecture. The C++ backend handles bidirectional radio I/O, multi-channel sensing, and real-time baseband processing, while the Python interaction layer provides visualization, experiment control, recording, and custom sensing algorithms. This preserves real-time performance while allowing experimenters to observe or extend the system without changing the radio processing path.

## Responsibilities of the two layers

- **Real-time layer:** performs uplink/downlink OFDM modulation and demodulation, synchronization, communication coding/decoding, and multi-channel sensing while keeping all USRP streams continuous.
- **Interaction layer:** displays multi-channel monostatic sensing, bistatic sensing, and communication-link status; adjusts parameters that support online updates; and receives data for further analysis.

Because of this separation, closing or replacing a visualization interface does not interrupt the backend-managed radio link or baseband pipeline. A Python frontend can run on the same host or connect over the network to a remote BS or UE.

## Data and control paths

Communication payloads and interaction data use separate transport paths:

- **Communication data plane:** downlink payloads enter the BS and leave the UE over UDP; uplink payloads enter the UE and leave the BS over UDP.
- **Sensing and status plane:** aggregated multi-channel monostatic data, bistatic results, uplink channel/delay-spectrum data, and runtime status are published over ZeroMQ.
- **Control plane:** Python uses ZeroMQ to query or update runtime parameters, including per-channel gain, alignment, and sensing settings that support online adjustment.

The BS aggregates multi-channel sensing by common frame-start symbol index and includes channel count, channel identifiers, and equal-sized channel payloads in one message. This preserves cross-channel timing. Sensing output still supports backend real-time processing and Python bypass, allowing users to select data granularity based on transport bandwidth, update-rate requirements, and the stage of algorithm validation.

## Benefits for experiments

Radio timing, TDD/FDD scheduling, multi-stream I/O, and buffer management remain in the real-time layer, so plotting or interaction cannot directly stall the baseband pipeline. The Python layer can evolve independently. The same BS/UE backend can therefore support live bidirectional communication and multi-channel sensing demonstrations as well as online data acquisition for new array-processing or ISAC algorithms.
