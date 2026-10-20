---
title: Hardware Setup
description: Hardware roles and basic preparation for OpenISAC experiments.
---

## Backend (C++)

To set up the complete system, you will need the following hardware:

*   **USRP Devices**: 2 units (e.g., USRP X310, B210, etc.)
*   **Computers**: 2 units (High performance recommended for signal processing)
*   **Antennas**: 3 for downlink-only operation; 5 if duplex/uplink is enabled. The BS uplink RX path currently cannot share the same RF channel as the sensing RX path.
*   **OCXO/GPSDO**: 2 units (Required for both USRPs)

### Connection Setup

The system consists of two main nodes:

1.  **BS Node**
    *   **Hardware**: 1x Computer, 1x USRP.
    *   **Antennas**: Connect 2 antennas for downlink-only operation (1 for downlink TX, 1 for sensing RX). If duplex/uplink is enabled, add a separate BS uplink RX antenna/RF chain; it cannot currently share the sensing RX channel.
    *   **Clock**: Connect an OCXO or GPSDO to the REFIN port of the USRP.
    *   **Function**: Transmits the OFDM signal and receives the radar echo.

2.  **UE Node**
    *   **Hardware**: 1x Computer, 1x USRP.
    *   **Antennas**: Connect 1 antenna to the RX port for downlink-only operation. If duplex/uplink is enabled, also connect the UE TX antenna/RF chain; in FDD mode, ensure the configured uplink carrier is supported.
    *   **Clock**: Connect an OCXO or GPSDO to the REFIN port of the USRP.
    *   **High-precision DAC (Optional)**: Use a high-precision DAC to enable finetuning the OCXO.
    *   **Function**: Receives the OFDM signal for communication and bistatic sensing; when duplex/uplink is enabled, also transmits UE->BS uplink payloads.

### Interface Requirements
To support high bandwidth and sample rates, ensure the connection between the Computers and USRPs uses:
*   **>= 10 Gigabit Ethernet (10GbE)** (For X-series)
*   **USB 3.0** (For B-series)

## Frontend (Python)
*   **Computer**: 1x Computer (Windows or Linux).
    *   Can be one of the backend computers or a separate machine.
*   **CPU**: High performance CPU (i7 10700 or better) if no GPU is available.
*   **GPU**: An Nvidia GPU is recommended for acceleration.
