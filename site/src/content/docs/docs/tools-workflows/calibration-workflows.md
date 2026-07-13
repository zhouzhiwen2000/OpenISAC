---
title: Calibration Workflows
description: Practical calibration sequence for sensing and timing experiments.
---

Calibration reduces ambiguity between propagation effects and hardware effects. Run system-delay calibration before `Calibrate Hsys`, because response calibration assumes the sensing RX frame is already aligned to the direct-path timing reference.

## Calibrate System Delay

First reduce transmit power for the direct connection. Lower `downlink.tx_gain` in `build/BS.yaml` and, if needed, insert a suitable attenuator so the sensing RX path is not saturated. Then connect the transmit RF output directly to the sensing RX input for the channel being measured, and keep that cable path stable during the test.

Enable system-delay estimation for only the sensing channel being measured:

```yaml
sensing:
  rx_channels:
    - usrp_channel: 1
      alignment: 63
      enable_system_delay_estimation: true
```

Start the BS backend from the `build/` directory. In this mode, the selected sensing channel disables the normal sensing pipeline and periodically runs a ZC-based delay test. Watch the BS console for `[SYSDLY CH <n>]` on CPU builds or `[CUDA SYSDLY CH <n>]` on CUDA builds. The CPU log prints `alignment_suggest=<value>`; the CUDA log prints `suggest=<value>`.

When the suggested value is stable, stop the backend, write that value back to the same channel's `alignment` field in `build/BS.yaml`, and turn the estimation mode off:

```yaml
sensing:
  rx_channels:
    - usrp_channel: 1
      alignment: <suggested value>
      enable_system_delay_estimation: false
```

Repeat the same direct-connection measurement for every sensing channel. For multichannel setups, move the RF direct connection to the next sensing RX path and update that channel's own `alignment`; do not reuse one channel's value for another RF path.

## Calibrate Sensing Channel Response

Before calibration, connect the sensing RF path directly: connect the transmit RF output to the corresponding sensing RX input and keep the connection stable during calibration.

If the transmit power is high, reduce the transmit power first. If needed, insert a suitable attenuator in the direct RF path to avoid RX saturation. The attenuator should be as flat as possible across the signal bandwidth; otherwise its in-band ripple will be included in the calibration result.

For monostatic sensing, start the BS backend and the monostatic sensing frontend, select the sensing channel you want to calibrate in the viewer, then click `Calibrate Hsys`. For multichannel monostatic setups, repeat this for each channel that needs its own RF path calibration.

For bistatic sensing, start the BS and UE backends, open the bistatic sensing frontend, then click `Calibrate Hsys` in the bistatic viewer while the RF path is directly connected.

Wait for the backend log to report that calibration has completed and the calibration file has been saved. The current run will immediately use the new response calibration. On later launches, the backend automatically loads the matching calibration file; if no matching file is found, sensing continues without calibration and prints a notice.

After calibration is complete, restore the normal antenna or experiment connection before continuing OTA measurements.
