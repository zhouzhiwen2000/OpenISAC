---
title: Sensing Viewer Guide
description: Screenshots, controls, parameter adjustments, and experiment workflows for the monostatic and bistatic sensing viewers.
---

This guide covers `scripts/plot_sensing_fast.py` and `scripts/plot_bi_sensing_fast.py` as of October 15, 2026. It is intended for users with a background in radar, wireless communications, or signal processing.

The viewers connect to a running OpenISAC system to display range or delay versus Doppler, observe micro-motion, extract targets, and adjust sensing and RF parameters.

Screenshots were generated offline from the viewer code, using illustrative spectra and readings.

## Interface purpose

| Item | Monostatic viewer | Bistatic viewer |
| --- | --- | --- |
| Launch script | `plot_sensing_fast.py` | `plot_bi_sensing_fast.py` |
| Connection destination | Computer running BS | Computer running UE |
| Main observation | Echoes of the base station's signal reflected by targets | Propagation paths and target scattering with separate transmit and receive sites |
| Main plot vertical axis | Range in m | Relative delay in ns |
| Main plot horizontal axis | Doppler frequency in Hz, or optional radial velocity in m/s | Doppler frequency in Hz |
| Micro-Doppler and detection | Micro-Doppler, OS-CFAR, CLEAN | Micro-Doppler, OS-CFAR, CLEAN |
| Delay estimation | FFT, MUSIC, CAPON, ROOTMUSIC, ESPRIT | FFT |
| RF gain adjustment | TX/RX gain controls in the viewer | BS/UE configuration |
| Hsys system-response calibration | Monostatic transmit/receive path | BS-to-UE sensing path |

Monostatic range uses the round-trip propagation relationship, with its origin established by system-delay calibration. Bistatic relative delay uses the current synchronization, compensation, and manual alignment settings as its reference. With direct-path tracking, it typically shows the additional delay of a scattered path relative to the direct path. Doppler describes changes in propagation-path length over time; interpret it using the corresponding monostatic or bistatic geometry.

## Startup and connection

![Backend IP, Connect, and connection status](/images/sensing-viewers/connection.webp)

Start the corresponding BS/UE backend and enable sensing output, then launch the viewer from the repository root:

```bash
# Monostatic sensing
python scripts/plot_sensing_fast.py

# Bistatic sensing
python scripts/plot_bi_sensing_fast.py
```

Use `python3` if that is the Python command in your environment. Enter `127.0.0.1` for a backend on the same computer, or the backend computer's IP address for a remote connection.

| Control or action | Function and operation |
| --- | --- |
| `Backend IP` + `Connect` | Enter the address of the computer running BS or UE, then connect or reconnect. Pressing Enter in the address field also connects. Wait for status and plot updates before observing results. |
| Arrow beside a section heading | Expand or collapse `Display & View`, `Delay Estimation`, `Target Detection`, `Clustering`, `Alignment`, `Hardware & RF`, or `Processing Toggles`. Enabled functions continue while their sections are collapsed. |
| Main window close button | Exit the viewer and close its target window. The independently running BS/UE backend continues. |

The viewer remembers several inputs and switches. After a successful connection or backend reconnection, it restores saved controls such as MTI and STRD; the monostatic viewer also sends its saved gain settings. Check the gains before connecting, particularly during RF loopback calibration.

## Interface overview

### Monostatic viewer

![Monostatic viewer with range-Doppler and micro-Doppler plots and control sections](/images/sensing-viewers/monostatic-overview.webp)

### Bistatic viewer

![Bistatic viewer with relative-delay-Doppler and micro-Doppler plots and control sections](/images/sensing-viewers/bistatic-overview.webp)

### Main spectrum

`Range-Doppler Spectrum` displays the joint distribution of range or delay and Doppler. Bright regions indicate stronger spectral responses. With detection enabled, markers identify extracted targets. The area near zero Doppler commonly contains static reflections, direct leakage, and slow targets. Doppler sign indicates frequency-shift direction; relate it to motion using the experiment geometry.

Click the main plot to mark an observation position with a cross. Select the micro-Doppler observation cell separately through `Range Bin`.

### Micro-Doppler plot

`Micro-Doppler Spectrum` shows Doppler variation over time at the selected range cell. The horizontal axis is time in seconds; the vertical axis is Doppler frequency in Hz. Use it to observe arm swing, limb motion, rotation, and other micro-motion. Select a new cell when a target moves across range cells.

FFT color scales use the current sensing spectrum's dB scale to compare relative component strength. MUSIC/CAPON use relative pseudospectrum scales to compare peak locations and separation.

### Status area

| Status | Meaning and use |
| --- | --- |
| `Display` | Source of the displayed data. |
| `FPS` | Viewer refresh rate, indicating display smoothness. |
| `Queue` | Data waiting for processing or display. Persistent accumulation with delayed plots suggests reducing local processing load. |
| `Sender`, `Status` | Data source, connection, and reception status. |
| `Params` | Current data and processing parameters; check after connecting or changing STRD. |
| `MD Buffer` | Accumulated micro-Doppler history. History starts again after selecting a different range cell. |
| `Detector` | Detection method, enabled state, and result summary. |

Hover over long status text for the full tooltip. Confirm device-control actions using new data, reported parameters, and backend logs.

## Display and observation range

![Display & View settings](/images/sensing-viewers/display-view.webp)

`Display & View` controls the observation cell, processing grids, visible region, and color scales. After editing a value, click `Set` or `Apply` on that row.

| Control or button | Function | Usage notes |
| --- | --- | --- |
| `Range Bin` + `Set` | Select the range or delay cell for micro-Doppler observation, indexed from 0. | Locate the target first, then select the cell. Applying the setting clears micro-Doppler history; allow it to accumulate again. |
| `Delay FFT` + `Set` | Set the viewer's range or delay transform length. | More zero padding produces a finer spectral grid for reading peaks and increases processing load. Physical range resolution depends on effective bandwidth and other acquisition conditions. The effective transform length covers at least the input data. |
| `Doppler FFT` + `Set` | Set the Doppler transform length. | More zero padding refines the frequency grid. Doppler resolving ability depends on coherent observation time and other acquisition conditions. |
| `Delay View` + `Set` | Set the number of range or delay bins displayed from the origin. | Focus on the near region or extend the view. The local target list also covers this region. Range Bin may be adjusted when the visible region shrinks. |
| `Doppler View` + `Set` | Set the number of bins displayed around zero Doppler. | A smaller view emphasizes slow targets; a larger view includes greater frequency shifts. Local target output follows the selected region. |
| `Delay Doppler dB` Min, Max + `Apply` | Set the main plot's color limits. | Lowering the minimum reveals weaker background; lowering the maximum makes weak responses brighter. These controls mainly apply to FFT; super-resolution methods use their own color scales. |
| `MicroDoppler dB` Min, Max + `Apply` | Set the micro-Doppler plot's color limits independently. | Emphasize weak motion patterns. Set the minimum below the maximum. |

Color-scale adjustments affect the display. Use detection parameters to adjust target selection, and RX gain to adjust received signal amplitude.

### Bins, range, delay, and frequency

Plot axes show m, ns, and Hz; bin inputs select discrete grid indices. Changing the FFT length changes the physical position represented by an index, so check the observation cell again.

The bistatic FFT display currently shows one point for every two range and Doppler cells. `Range Bin` selects a micro-Doppler cell on the full range grid. For example, a target listed at `R=40` can be located starting at `Range Bin=80`. The monostatic display uses the full grid by default. Plot axes already account for this conversion.

After changing FFT lengths or display ranges, sensing results and micro-Doppler history are rebuilt. Wait for new data to stabilize before continuing.

## Sensing processing switches

![Micro-Doppler, MTI, and window switches](/images/sensing-viewers/processing-toggles.webp)

`Processing Toggles` controls micro-Doppler, moving-target indication, and window functions. MTI and the two window buttons are green when enabled.

| Button | Effect when enabled | Typical use |
| --- | --- | --- |
| `Micro-Doppler: ON/OFF` | Start or stop micro-Doppler history accumulation and processing. | Enable for human or mechanical micro-motion. Disable when observing only the main spectrum to reduce processing load. The retained image after disabling shows earlier results. |
| `MTI` | Control backend moving-target indication to suppress static or slowly varying background. | Reduce static clutter for moving-target detection. Compare with MTI disabled when studying stationary reflections, direct paths, or slow motion. |
| `Range Window` | Apply a Hamming window in range processing to reduce range sidelobes. | Useful when strong-return sidelobes obscure weaker returns. Windowing also broadens the main lobe, affecting separation of nearby targets. |
| `Doppler Window` | Apply a Hamming window in Doppler processing to reduce spectral leakage and Doppler sidelobes. | Useful for weaker motion components, while accounting for main-lobe broadening. |

Selecting CLEAN in FFT mode automatically turns off and disables both Window buttons. After switching back to OS-CFAR, enable them as needed.

## Target detection

### Method and controls

| Control or button | Function and operation |
| --- | --- |
| `Local Detector` dropdown | Select `OS-CFAR` or `CLEAN` for local detection. The method changes immediately. |
| `Local OS-CFAR: ON/OFF` | Enable or disable OS-CFAR. When enabled, it estimates a local background threshold and produces target markers and a list. |
| `Local CLEAN: ON/OFF` | The same switch uses this label when CLEAN is selected. It extracts strong targets and subtracts their responses to find remaining components. |
| `Apply OS-CFAR` | Apply the current OS-CFAR parameters. Confirm that detection is ON after applying. |
| `Apply CLEAN` | Apply the current CLEAN parameters. Confirm that detection is ON after applying. |

OS-CFAR extracts significant returns relative to local noise and clutter. CLEAN helps examine additional components affected by strong targets and their sidelobes. Markers indicate candidate scattering targets in the current frame; assess them using position, stability, and the experiment scene.

When the monostatic viewer switches to a non-FFT method such as MUSIC, that method supplies targets directly and the regular detection controls are hidden. Switching back to FFT restores them.

### OS-CFAR parameters

![OS-CFAR switch and parameters](/images/sensing-viewers/os-cfar.webp)

`D` denotes Doppler; `R` denotes range or delay. Window sizes are expressed in bins on the current detection grid.

| Parameter | Meaning | Adjustment effect |
| --- | --- | --- |
| `Rank(%)` | Percentile of sorted reference-cell powers used to estimate the background. | With other settings fixed, raising the rank typically raises the background estimate and threshold, making detection more conservative. |
| `Train D`, `Train R` | Reference-region widths along Doppler and range. | Small windows are more sensitive to local fluctuations; large windows can include other targets or different clutter regions. |
| `Guard D`, `Guard R` | Half-widths of the protected region around a target, excluded from background estimation. | Cover the region around the main lobe to reduce target-energy contamination of the background estimate. |
| `Supp D`, `Supp R` | Half-widths of the region where repeated reports are suppressed after accepting a peak. | Larger values reduce duplicate points but can suppress nearby real targets. They also set the DBSCAN neighborhood. |
| `Alpha(dB)` | Threshold offset above the estimated background. | Increasing it reduces weak detections and false alarms; decreasing it increases sensitivity to weak targets. |
| `Min R` | Minimum range-bin index searched for targets. | Exclude near-end leakage or an unwanted near region. |
| `Min P(dB)` | Minimum spectral strength for candidate points. | Uses the current spectrum's scale and works together with the local adaptive threshold. |
| `DC Excl` | Half-width of the excluded region around zero Doppler. A value of 0 retains that region. | Larger values reduce static residuals as well as detections of stationary and slow targets. |

Start with `Alpha(dB)` and `Min P(dB)` to obtain stable targets, then adjust guard, suppression, and reference regions according to sidelobe width and target spacing. Detection near plot boundaries requires enough reference cells.

### CLEAN parameters

![CLEAN switch and parameters](/images/sensing-viewers/clean.webp)

| Parameter | Meaning | Adjustment effect |
| --- | --- | --- |
| `Loop Gain` | Fraction of the selected target response subtracted each iteration, from 0.001 to 1. | Smaller values leave more residual response and can produce nearby candidate components. Larger values subtract more fully and are more sensitive to model error. |
| `Max Targets` | Maximum number of target components extracted in one processing pass. | Set the extraction limit according to scene complexity. |
| `Min R` | Exclude range cells below this index. | Avoid strong components such as leakage. |
| `Min P(dB)` | Stop extracting when the remaining peak falls below this strength. | Lower values extract more weak components and may include noise or residual sidelobes. |
| `DC Excl` | Half-width of the excluded region around zero Doppler. | Set to 0 for static targets and compare with MTI disabled. |

Both methods share the current `Min R` and `DC Excl` limits. Check these two settings after switching methods.

## Target clustering

![DBSCAN neighborhood, MinPts, and Apply DBSCAN](/images/sensing-viewers/clustering.webp)

`Clustering` organizes OS-CFAR detection points. A scattering target can produce several adjacent points; clustering merges these into target entries to reduce duplicate reports.

| Control or button | Function and operation |
| --- | --- |
| `DBSCAN Eps` | Display the Doppler and range neighborhood sizes, respectively `Supp D + 1` and `Supp R + 1`. Change them by adjusting and applying the OS-CFAR suppression settings. |
| `MinPts` + `Apply DBSCAN` | Set the neighboring-point count required for a dense region. Increasing it helps reject isolated points and may reduce weak-target detections. A value of 1 permits a single point to form a target. |

DBSCAN applies to OS-CFAR results. CLEAN and super-resolution methods output targets directly; selecting a super-resolution method hides the clustering controls.

## Monostatic delay estimation

![Delay Estimator selection and Peak Rel threshold](/images/sensing-viewers/delay-estimation.webp)

`Delay Estimation` changes the estimator along range or delay to compare the separation of nearby paths or targets.

| `Delay Estimator` option | Result and use |
| --- | --- |
| `FFT` | Conventional range-Doppler spectrum for a baseline, scene overview, and OS-CFAR or CLEAN detection. |
| `MUSIC` | Delay pseudospectrum and peaks for examining nearby paths. Results depend on SNR, data conditions, and model error. |
| `CAPON` | Adaptive delay spectrum for comparing nearby-component separation and interference effects. |
| `ROOTMUSIC` | Delay candidates, presented primarily as target points and list entries. |
| `ESPRIT` | Direct delay estimates for comparison with FFT and MUSIC results. |

`Peak Rel(dB)` + `Apply` sets the candidate-peak threshold relative to the strongest peak in the current result. For example, `-12` retains candidates within about 12 dB of the strongest peak. `-6` is more selective; `-20` retains more weak peaks.

Wait for recalculation after changing methods. Assess super-resolution results against known target positions, peak stability, and the FFT baseline.

## Delay alignment

![Alignment increment input and adjustment buttons](/images/sensing-viewers/alignment.webp)

`Alignment` adjusts the sensing delay reference. Values are in original signal samples, and every button applies an increment.

| Button | Function |
| --- | --- |
| `Delay` + `Apply` | Apply the signed integer as the next alignment increment. Entering 10 and clicking twice applies two successive +10 adjustments. |
| `+57600` | Adjust by +57600 samples for coarse correction of large offsets. |
| `-57600` | Adjust by -57600 samples. |
| `+10` | Adjust by +10 samples. |
| `-10` | Adjust by -10 samples. |
| `+1` | Adjust by +1 sample for fine correction. |
| `-1` | Adjust by -1 sample. |

The monostatic viewer adjusts sensing receive alignment; the bistatic viewer adds a manual delay correction to UE sensing. Select a clear reference peak, use +1 or -1 to observe the direction of change, then choose a suitable step.

Alignment directly affects the range or delay reference. Record the final alignment conditions and save suitable calibration values in the system configuration to reproduce the experiment.

## System control and calibration

### Sampling stride and gain

![Hardware & RF sampling stride, gains, and Hsys calibration](/images/sensing-viewers/hardware-rf.webp)

| Control or button | Function | Usage notes |
| --- | --- | --- |
| `STRD` + `Set` | Set the OFDM-symbol interval used for sensing slow-time samples. | For a fixed sample count, a larger stride spans more time but reduces the slow-time sampling rate and unambiguous Doppler interval. Smaller strides suit faster motion. |
| `Frame every N` + `Set` | Process/output one complete sensing frame out of N; `1` keeps all. | Preserves within-frame STRD and unambiguous Doppler interval; reduces refresh rate and downstream load. Micro-Doppler history restarts across gaps; use `1` for continuous analysis. |
| `TX Gain(dB)` + `Set` | Change BS transmit gain; available in the monostatic viewer. | Changes the transmitted signal and affects both sensing echoes and the communication link. |
| `RX Gain(dB)` + `Set` | Change monostatic sensing receive gain. | Balance weak-target visibility with the receiver's linear operating range. |

The system configuration determines valid STRD values. With `compact_mask` resource patterns, sampling locations follow the pattern and are adjusted in configuration. Confirm changes using reported parameters and logs. Bistatic TX/RX gains are set in the corresponding BS/UE configurations.

### Hsys system-response calibration

`Calibrate Hsys` requests a measurement and compensation of the transmit/receive frequency response to reduce hardware amplitude and phase variations in the sensing spectrum. Monostatic calibration covers its transmit/receive path; bistatic calibration covers the BS-to-UE sensing path.

1. Complete system-delay alignment, set suitable TX/RX gains, and prepare enough external attenuation to keep the receiver in its linear operating range.
2. Connect the transmit and receive RF ports for the calibration path. For bistatic calibration, connect BS transmit to UE receive and run both backends.
3. Click `Calibrate Hsys` and keep the connections and gains stable.
4. Wait for backend logs to confirm completion and file saving.
5. Stop the relevant backends before restoring antennas or experiment connections, then resume over-the-air measurements.

The current session uses the new calibration; later sessions load a matching file. Reassess calibration after changing gain, cables, or other RF conditions. Perform Hsys calibration with the prepared RF loopback connection. See [Calibration Workflows](https://openisac.zzw123app.top/docs/tools-workflows/calibration-workflows/) for the complete preparation and verification procedure.

## Target list

![OpenISAC Top Targets list](/images/sensing-viewers/targets.webp)

`OpenISAC Top Targets` lists up to 12 entries, sorted by target strength in the current frame. Entry numbers update with each frame's ordering.

| Field | Meaning |
| --- | --- |
| `R` | Range or delay index on the current display grid. |
| `rel` | Range or delay relative to the current reference, expressed in m or ns when physical parameters are available. |
| `f_D` / `v_r` (monostatic), `D` (bistatic) | Monostatic target frequency in Hz or radial velocity in m/s, matching the plot. Bistatic `D` remains a bin index relative to zero Doppler. |
| `S` | Target strength, interpreted using the selected method's spectral scale. |
| `N` | Number of detection points in the target entry. |
| `C` | Range and Doppler indices of the detection cluster's center. |

Use the list to inspect target position, strength, and cluster distribution. Compare the stability of nearby results across frames and with the plot when assessing detection thresholds and clustering settings.

## Saving data

![Save Raw, Save RD, and Save MD buttons](/images/sensing-viewers/save-data.webp)

Each button saves one snapshot in MATLAB `.mat` format, which can also be read with Python.

| Button | Saved content | Use |
| --- | --- | --- |
| `Save Raw` | Sensing input frame received by the viewer, usually complex channel data, in `raw_frame`. | Reprocess offline or compare algorithms. The system output mode determines the data's processing stage. |
| `Save RD` | Current range/delay-Doppler display matrix in `rd_map`. | Retain the current spectrum, including display cropping and algorithm effects. ROOTMUSIC/ESPRIT primarily produce target points; record their point lists separately. |
| `Save MD` | Accumulated micro-Doppler display matrix in `md_map`. | Save the motion time-frequency result after confirming the plot has updated. |

Files are written to `capture/` under the directory from which the viewer was launched, for example `capture/capture_rd_20261015_143000.mat`. The terminal displays the saved path.

Record FFT lengths, STRD, algorithm, observation cell, and other experiment conditions to restore range, delay, and frequency units and reproduce results. Filenames have one-second time resolution, so space saves of the same type by at least one second. When comparing data from the same frame, also check acquisition times.

## Typical workflows

### Moving targets

1. Connect to BS or UE, confirm continuous main-plot updates, and establish an FFT baseline.
2. Enable MTI. With OS-CFAR, enable the window functions as needed and observe sidelobes and main-lobe width.
3. Set Delay View, Doppler View, and color limits to show the target clearly.
4. Enable OS-CFAR, adjust Alpha and Min P, then tune suppression and clustering to reduce duplicate points.
5. Check target-list positions, strengths, and stability; save RD and record the settings.

### Static paths

1. Disable MTI and apply a DC Excl value of 0.
2. Check Min R so the reference path and near targets fall inside the search region.
3. Observe the spectrum near zero Doppler, then enable and tune detection.
4. For bistatic experiments, record the delay-compensation method and manual alignment offset when comparing path changes.

### Micro-Doppler

1. Locate the target's range or delay on the main plot.
2. Enter its full-grid range index in Range Bin and click Set; apply the ratio described above for the bistatic viewer.
3. Enable Micro-Doppler, wait for history to accumulate, and adjust color limits to observe motion patterns.
4. Select a new cell when the target crosses range cells, then allow new history to build.
5. Click Save MD and record the observation cell, STRD, and sampling conditions.

## Runtime checks

| Observation | Items to check |
| --- | --- |
| No spectrum after connecting | Backend computer address, BS/UE status, sensing-output setting, data mode, and terminal errors. |
| Target visible in the spectrum but list empty | Detection switch, Alpha and Min P, Min R and DC Excl, observation range, and OS-CFAR MinPts. |
| Static reflection or direct path disappears | Compare with MTI disabled and check the zero-Doppler exclusion region. |
| Micro-Doppler blank or showing an old image | Micro-Doppler switch, history accumulation, Range Bin, and recent parameter changes. |
| Both Window buttons disabled | Check whether CLEAN is selected; switch to OS-CFAR to enable the windows as needed. |
| Some controls hidden | Viewer type, FFT/super-resolution selection, and expanded or collapsed sections. |
| Larger FFT still gives poor target separation | Effective bandwidth, coherent observation time, SNR, windowing, and target spacing. Zero padding refines the grid. |
| STRD change has no visible effect | Resource-pattern sampling and valid configuration values; confirm reported parameters and logs. |
| Previous settings restored after restarting | Remembered inputs and switches; apply edits with Set/Apply and check the current session. |

## Related documentation

- [Scripts and Tools](https://openisac.zzw123app.top/docs/reference/scripts-tools/)
- [System Delay and Hsys Calibration](https://openisac.zzw123app.top/docs/tools-workflows/calibration-workflows/)
- [Monostatic Sensing](https://openisac.zzw123app.top/docs/signal-processing/monostatic-sensing/)
- [Bistatic Sensing](https://openisac.zzw123app.top/docs/signal-processing/bistatic-sensing/)
- [Troubleshooting](https://openisac.zzw123app.top/docs/reference/troubleshooting/)

### Optional radial velocity in the monostatic viewer

In `plot_sensing_fast.py`, open **Display & View**, enter the actual radiated carrier in **Velocity RF Carrier (GHz)**, and click **Set** (or press Enter). The range-spectrum horizontal axis, micro-Doppler vertical axis, and detected-target list then show radial velocity in **m/s**. Clear the field and apply it to return to Doppler frequency in **Hz**. The applied value is saved for the next launch.

The conversion is $v_r = f_D c / (2 f_c)$, with the sign inherited from the displayed Doppler frequency. For an external millimetre-wave converter, enter its RF output carrier rather than the USRP intermediate frequency; for example, enter `77` only when the actual RF carrier is 77 GHz. This display setting does not tune the radio or change FFT processing, detector thresholds, or bin coordinates. Positive/negative motion direction must be checked against the converter and experiment's frequency-sign convention. The bistatic viewer keeps its Doppler-frequency display because radial speed cannot be inferred with the monostatic formula.

A separate yellow annotation at the top right of the main spectrum shows the strongest detected return's range and velocity to one decimal place (or Doppler frequency when the carrier is blank). The annotation keeps a fixed position and size; missing target values are shown as **N/A**.

The monostatic fast viewer keeps its rolling micro-Doppler history across dropped or decimated frames and inserts zeros at missing slow-time positions. This preserves elapsed time; it does not reconstruct missing echoes, and gaps can weaken peaks or introduce sidelobes. History restarts when the range cell or sampling geometry changes, or the timestamp moves backwards.
