---
title: UE Runtime Pipeline
description: UE data flow for downlink reception, bistatic sensing, and uplink transmission.
---

The UE uses concurrent FIFO pipelines for downlink reception, bistatic sensing, and uplink transmission. Communication and sensing share the downlink demodulation results, while the uplink has its own coding, modulation, and transmit path.

![UE bidirectional communication and bistatic sensing software architecture](/images/SoftArchUEDuplex.png)

## Reception and synchronization

1. The USRP continuously captures the BS downlink and passes samples through the receive queue to OFDM demodulation.
2. During initial acquisition, known synchronization resources locate the frame boundary and estimate frequency offset. After acquisition, the receiver processes the stream frame by frame.
3. Carrier-frequency offset, sampling-clock mismatch, and timing drift are tracked during operation to keep OFDM symbols aligned.
4. Cyclic-prefix removal, FFT, channel estimation, and equalization convert the waveform into frequency-domain symbols shared by communication and sensing.

## Parallel communication and sensing branches

The demodulated data feeds two concurrent branches:

- **Communication:** soft information enters LDPC decoding, and recovered payloads are forwarded to applications over UDP.
- **Bistatic sensing:** the UE reconstructs transmitted data symbols from equalized decisions and directly uses known synchronization and pilot symbols. Paired receive symbols and reconstructed transmit references form the bistatic time-frequency channel and delay-Doppler output.

Reusing the demodulation results avoids duplicating the full receive chain for sensing and ensures that communication and bistatic sensing refer to the same frame and channel observation.

## UE-to-BS uplink transmission

1. The UE receives uplink payloads on a separate UDP input and performs packet framing, scrambling, and LDPC encoding.
2. The OFDM modulator maps uplink synchronization resources, pilots, and payloads onto the resource grid and generates a time-domain waveform.
3. In TDD, the UE transmits only in the configured in-frame uplink window after its guard symbols. In FDD, it continuously transmits full uplink frames on a separate carrier.
4. A transmit queue delivers the generated waveform to the USRP, and timing advance aligns the uplink transmission with the BS receive window.

The uplink uses the same FFT, cyclic-prefix, and pilot framework as the downlink. When no application payload is available, a configurable idle waveform provides deterministic transmit behavior.

## Why synchronization serves both functions

Communication mainly requires stable frame boundaries and controlled inter-symbol interference. Bistatic sensing must also preserve continuous delay and phase evolution, making it more sensitive to drift between the independent BS and UE clocks. The UE therefore supplies demodulator-derived frequency and timing estimates to the sensing path and uses an OTA reference path to compensate sensing-side delay drift.

For experiments that need longer coherent observation, the optional hardware synchronization controller can trim the UE local clock to reduce long-term carrier and sampling-clock offsets. Bistatic sensing can produce results directly in the backend or export channel data to Python for custom analysis.

In most OTA runs, start the UE before the BS so it is ready to acquire synchronization when the downlink begins. When checking bidirectional operation, confirm downlink synchronization and decoding first, then the uplink receive window, and finally the stability of bistatic sensing output.
