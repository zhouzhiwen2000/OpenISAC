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

With `ofdm_frame.enable_sec_sync_symbol` enabled, the full-frame correlator can select either of the two identical ZC sync peaks, producing suggestions one complete OFDM symbol (`fft_size + cp_length` samples) apart. If this occurs, stop BS and UE, temporarily set this flag to `false` in BS, and run only BS for delay estimation. After applying the alignment, repeat estimation to check that the residual delay is near zero. Stop BS and restore the duplicate sync symbol before Hsys calibration and communication.

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

Before starting the backend or connecting the RF loopback, set suitable gains and choose sufficient external attenuation to keep the RX path linear. Then connect the transmit RF output to the corresponding sensing RX input and keep that path stable during calibration. The attenuator should be as flat as possible across the signal bandwidth; otherwise its in-band ripple will be included in the calibration result.

For monostatic sensing, start the BS backend and the monostatic sensing frontend, select the sensing channel you want to calibrate in the viewer, then click `Calibrate Hsys`. For multichannel monostatic setups, repeat this for each channel that needs its own RF path calibration.

For bistatic sensing, start the BS and UE backends, open the bistatic sensing frontend, then click `Calibrate Hsys` in the bistatic viewer while the RF path is directly connected.

Wait for the backend log to report that calibration has completed and the calibration file has been saved. The current run will immediately use the new response calibration. On later launches, the backend automatically loads the matching calibration file; if no matching file is found, sensing continues without calibration and prints a notice.

After calibration is complete, stop the backend(s), restore the normal antenna or experiment connection, and then resume OTA measurements. Reassess calibration if gains or other RF settings change.


## Verify Hsys from the command line

As an alternative to clicking the calibration button in the sensing viewer, you can calibrate Hsys from the command line. Choose either method. First follow the preparation steps above and start the backend: BS for monostatic sensing, or both BS and UE for bistatic sensing. Then open another terminal at the repository root and run the appropriate command below.

For the command-line method, you do not need to launch the sensing viewer. The script connects to an already running backend; it does not start the backend or configure CPU isolation. If you use CPU isolation, set it up before starting the backend.

```bash
# BS monostatic logical channel 0 (not necessarily USRP channel 0)
python3 scripts/calibrate_hsys.py capture --backend-dir build --channel 0

# UE bistatic path; control port is read from build/UE.yaml
python3 scripts/calibrate_hsys.py capture --backend-dir build --role bistatic --channel 0

# Inspect an existing file without connecting to a backend
python3 scripts/calibrate_hsys.py inspect build/sensing_system_response_monostatic_ch0.bin
```

1. Complete system-delay alignment first and disable `enable_system_delay_estimation`. Use dense sensing or a full-band regular compact mask.
2. Set suitable TX/RX gains **before starting the backend and connecting the attenuated RF loopback**. Prefer the intended operating gains with sufficient external attenuation; gain changes can change the RF response. Ensure the receiver remains linear. A gain setting of 0 dB does **not** turn TX off.
3. Start the backend using its actual working-directory YAML. Point `--backend-dir` at that same directory. For a remote backend, use `--host` and a shared mount of its working directory. A local directory containing only a copied YAML is insufficient for completion verification.
4. Close other control clients during calibration. The existing `ALCH` selection is backend-global; concurrent viewer controls can redirect a request. The script explicitly selects one logical sensing channel and leaves that selection in place.
5. Run `capture` and confirm the prepared setup. `--yes` skips only this interactive confirmation. `--symbols` defaults to 1000, `--timeout` to 60 seconds, and `--connect-timeout` to 5 seconds. A large sensing stride or slow capture may require a longer timeout.
6. Wait for **Saved file verified**, then stop the backend before changing RF connections or restoring gains. Repeat separately for each RF path.

The tool requires V9 parameter replies and checks the backend role, FFT and RF geometry against the YAML. It backs up an existing calibration as a uniquely named `.bin.<id>.bak` file before requesting a new capture. Completion requires a newly written, complete Hsys file with matching role, channel, FFT, RF metadata and symbol count; a preexisting file or a successful command send is not treated as completion. The JSON summary includes a SHA-256 digest, magnitude percentiles and weak-bin count (the backend leaves bins below its inverse-response threshold unequalized).

The script does not change gains, rewrite YAML or restore files automatically. On timeout, interruption or validation failure it exits nonzero; backend capture may still be active, so stop the backend before changing wiring or gains. Keep the backup until the new calibration has been checked. The binary format is native-endian; inspect files on a host with the backend's byte order.

File validation proves structural and numerical consistency, not absence of clipping, adequate SNR or correct RF wiring. The current file header does not record device serials, gains, antenna ports or cable/attenuator identity. Recalibrate when those RF conditions change, even if the backend accepts the file. The loopback cable and attenuator response are included in Hsys; this is not an absolute range or antenna calibration.

The command-line workflow builds on [Henry Teall's calibration tooling](https://github.com/philippaPA/OpenISAC/commit/9b630f3744a2261b234d22f95bfe6e07b727d4d8). TX-off DC-baseline capture is not included.

Hsys uses the original OFDM FFT in `ofdm_frame.fft_size`; `sensing.range_fft_size` and the viewer's padded range/Doppler FFT lengths can differ. Record the TX/RX gains and attenuation separately when using temporary loopback gains. Stop the backend and restore the experiment wiring before restoring operating gains, then check the correction. A matching file header does not prove that the RF response is unchanged at different gains.

## Analyze system-delay logs

Save the backend output from one calibration run and analyze the recent estimates:

```bash
python3 scripts/calibrate_system_delay.py capture/system-delay.log --channel 0
# Also accepts a completed log on stdin
cat capture/system-delay.log | python3 scripts/calibrate_system_delay.py -
```

The tool parses CPU and CUDA/ROCm `SYSDLY` log formats and reports channels separately. By default it examines the latest 20 observations, requires at least 10, allows a maximum spread of 2 samples, and requires every peak/mean correlation ratio to be at least 10. These are configurable screening thresholds (`--window`, `--min-samples`, `--max-spread`, `--min-ratio`), not a statistical accuracy guarantee. A backend sequence restart or an alignment change resets the history; duplicate records do not count as additional measurements. Unstable or insufficient data yields no alignment recommendation and a nonzero exit code.

Review the resulting integer `alignment`, stop the backend, update that logical channel's YAML and disable estimation. The tool never edits configuration or sends hardware commands. Use logs from one backend run; suggested alignment includes the calibration path and does not measure absolute propagation delay.
