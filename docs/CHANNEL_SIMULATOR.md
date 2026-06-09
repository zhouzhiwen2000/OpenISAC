# Channel Simulator — run OpenISAC without a USRP

> 中文版见 [CHANNEL_SIMULATOR.zh-CN.md](CHANNEL_SIMULATOR.zh-CN.md)。


The `ChannelSimulator` lets you run the full communication **and** multi-channel
sensing pipeline with no radio attached. It replaces the USRP with a shared-memory
"air" that applies a configurable channel model: point targets with delay / Doppler /
gain / steering vector, AWGN, CFO, and timing offset, plus a comm tapped-delay-line
for the direct/LoS path.

Each sensing RX channel is treated as one antenna; the steering vector is either a
parametric uniform linear array (`array_spacing_lambda`) or a fully custom matrix
loaded from `steering_override_file`. The communication channel is a **realistic
multipath channel**: the LoS taps plus the same moving target reflections, so the
demodulator's **bistatic** sensing has detectable range-Doppler targets too.

## How it works

```
 OFDMModulator ──TX samples──▶ ChannelSimulator ──per-antenna RX──▶ OFDMDemodulator (comm RX)
 (SimTxStreamer)                (the "air" + clock)  └─sens RX×N──▶ OFDMModulator monostatic sensing
```

All three processes attach to POSIX shared-memory rings named by `simulation.session`.
The engines select the backend with `radio_backend: sim` in their YAML; the hot
`send()/recv()` paths are unchanged because the sim streamers implement the abstract
`uhd::tx_streamer` / `uhd::rx_streamer` interfaces. `radio_backend: uhd` (the default)
keeps the original hardware behaviour.

## Build

```
cd build && cmake .. && make -j ChannelSimulator OFDMModulator OFDMDemodulator
```

## Run (three terminals, same machine)

```
cd build
cp ../config/Modulator_Sim.yaml   Modulator.yaml
cp ../config/Demodulator_Sim.yaml Demodulator.yaml

# 1) start the "air" FIRST (it creates the shared memory and is the sample clock)
./ChannelSimulator                 # reads Modulator.yaml (+ its simulation: block)

# 2) start the transmitter / monostatic sensing
./OFDMModulator

# 3) start the receiver / comm demod
./OFDMDemodulator
```

Start `ChannelSimulator` first. By default the hub applies backpressure, so a path whose
consumer is absent will pause the simulation — run every enabled receiver. To run only one
side, disable the other with the flags below (the hub then neither creates nor produces that
path, so its consumer need not run):

- **Sensing only** — set `enable_comm_rx: false`; run `ChannelSimulator` + `OFDMModulator` (no demodulator).
- **Comm only** — set `enable_sensing_rx: false`; run all three (the modulator builds no sensing channels).

You can also disable sensing the legacy way by setting `sensing_rx_channel_count: 0`.

Visualize sensing with `python3 scripts/plot_sensing_fast.py` (RD streaming is started
by the viewer, exactly as with hardware).

## Configuring the channel (`simulation:` block in Modulator_Sim.yaml)

```yaml
radio_backend: sim
simulation:
  session: oisac_sim             # must match across all three processes
  enable_comm_rx: true           # produce the comm RX path (false = sensing-only, no demodulator)
  enable_sensing_rx: true        # produce the sensing RX paths (false = comm-only)
  noise_power_dbfs: -50          # AWGN per RX channel; <= -200 disables
  cfo_hz: 0.0                    # carrier frequency offset on RX
  timing_offset_samples: 0       # constant RX sample delay
  array_spacing_m: 0.04283       # physical ULA spacing (m); d/λ scales with center_freq
                                 # (42.83 mm = λ/2 @ 3.5 GHz, matches the sensing viewers).
                                 # Set <= 0 to fall back to the legacy array_spacing_lambda.
  array_spacing_lambda: 0.5      # legacy fixed spacing (wavelengths); used only if array_spacing_m <= 0
  ring_capacity_samples: 262144  # ~2 frames; small keeps TX close to the hub
  steering_override_file: ""     # [num_targets x num_channels] complex<float>, row-major; empty = ULA
  comm_multipath_taps:           # comm direct/LoS + static multipath (decodable component)
    - { delay_samples: 0, gain_db: 0, phase_deg: 0 }
  targets:                       # monostatic sensing scatterers (with steering)
    - { range_m: 30, velocity_mps: 5,  gain_db: -6,  angle_deg: 20 }
    - { range_m: 75, velocity_mps: -3, gain_db: -12, angle_deg: -10 }
  # bistatic_targets:            # optional independent scene for the bistatic (comm) channel
  #   - { range_m: 45, velocity_mps: 8, gain_db: -8, angle_deg: 0 }
```

The number of sensing antennas equals the number of `sensing_rx_channels` entries.

`targets` feeds the monostatic sensing antennas (with steering). The bistatic (comm)
channel uses `bistatic_targets` when it is non-empty, otherwise it falls back to
`targets`, so a single scene drives both views by default. `angle_deg` is ignored on
the bistatic channel (the comm RX is a single antenna).
A custom steering override file is `num_targets × num_channels` little-endian
`complex<float>` values in row-major order (target-major).

## Channel model

- Target round-trip delay `τ = 2·range_m/c`, sample delay `round(τ·fs) + timing_offset`.
- Doppler `fd = 2·velocity_mps/λ`, `λ = c/center_freq`.
- ULA steering `a_k(θ) = exp(j·2π·(d/λ)·k·sinθ)` for antenna `k`, where the electrical
  spacing `d/λ = array_spacing_m·center_freq/c` is derived from the physical spacing and
  the carrier (so the recovered angle is correct at any `center_freq`; the viewers invert
  the phase slope using the same physical spacing). With `array_spacing_m <= 0` the legacy
  frequency-independent `array_spacing_lambda` is used instead.
- Monostatic sensing RX (per antenna `k`):
  `rx_sens_k[n] = Σ_targets gain·a_k(θ)·tx[n−τ]·e^{j2π fd n/fs} + AWGN`.
- Comm / bistatic RX (single antenna): a decodable direct path **plus** the same moving
  target reflections, so the demodulator's bistatic sensing has range-Doppler targets:
  `rx_comm[n] = (Σ_taps gain·tx[n−delay] + Σ_targets gain·tx[n−τ]·e^{j2π fd n/fs})·e^{j2π cfo n/fs} + AWGN`.

By default the `targets` list drives both the monostatic sensing antennas (with steering) and the
bistatic comm channel (no steering), modelling one coherent scene. Set `bistatic_targets` to give
the comm/bistatic channel its own independent scatterers. Keep the direct-path
`comm_multipath_taps` stronger than the targets (e.g. 0 dB vs −6/−12 dB) so the comm link still
synchronizes and decodes.

## Notes / limits

- Each process stops cleanly on Ctrl-C (SIGINT), independently of the others; the hub
  unlinks its shared memory on exit.
- Not real-time: pacing is by shared-memory backpressure (correctness over speed).
- Timed TX bursts are approximated as a continuous stream; RX sync recovers framing.
- Keep `ring_capacity_samples` small (≈ a couple of frames) so the transmitter cannot
  race far ahead of the hub and overflow the monostatic TX/RX pairing queue.
```
