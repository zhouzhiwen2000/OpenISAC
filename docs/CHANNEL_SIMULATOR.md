# Channel Simulator — run OpenISAC without a USRP

> 中文版见 [CHANNEL_SIMULATOR.zh-CN.md](CHANNEL_SIMULATOR.zh-CN.md)。


The `ChannelSimulator` lets you run the full communication **and** multi-channel
sensing pipeline with no radio attached. It replaces the USRP with shared-memory
"air" and builds a scatterer-based multipath channel: the transmitted samples pass
through a LoS path, optional static multipath paths, and moving scatterers with
delay / Doppler / gain / array response. A fixed integer timing offset can be folded
into each path delay; the communication/bistatic receive chain then applies relative
carrier frequency offset (CFO), and AWGN is added to every RX output. The timing
offset here is a fixed integer sample offset; the simulator does not model sampling
frequency offset (SFO).

Each sensing RX channel is treated as one antenna. The array response can be generated
from a parametric uniform linear array (ULA, `array_spacing_lambda`) or loaded as a
custom array manifold matrix from `steering_override_file`. The communication channel
is a **realistic multipath channel**: a LoS path plus scatterer returns, so the
UE's **bistatic** sensing can also detect range-Doppler targets.

## How it works

```
 BS ──TX samples──▶ ChannelSimulator ──per-antenna RX──▶ UE (comm RX)
 (SimTxStreamer)                (the "air" + clock)  └─sens RX×N──▶ BS monostatic sensing
```

All three processes attach to POSIX shared-memory rings named by `simulation.session`.
The engines select the backend with `radio_backend: sim` in their YAML; the hot
`send()/recv()` paths are unchanged because the sim streamers implement the abstract
`uhd::tx_streamer` / `uhd::rx_streamer` interfaces. `radio_backend: uhd` (the default)
keeps the original hardware behaviour.

## Build

```
cd build && cmake .. && make -j ChannelSimulator BS UE
```

## Run (three terminals, same machine)

```
cd build
cp ../config/BS_Sim.yaml   BS.yaml
cp ../config/UE_Sim.yaml UE.yaml

# 1) start the "air" FIRST (it creates the shared memory and is the sample clock)
./ChannelSimulator                 # reads BS.yaml (+ its simulation: block)

# 2) start the transmitter / monostatic sensing
./BS

# 3) start the receiver / comm demod
./UE
```

Start `ChannelSimulator` first. By default the hub uses backpressure: if communication
RX or sensing RX is enabled but the corresponding consumer process is not reading from
shared memory, the related ring buffer will fill and the whole simulation will wait.
Start every enabled receiver; if you only need one side, disable the unused output in
the `simulation:` block:

- **Sensing only** — set `enable_comm_rx: false`; the hub does not produce communication RX output, so run only `ChannelSimulator` + `BS` and skip `UE`.
- **Comm only** — set `enable_sensing_rx: false`; the hub does not produce sensing RX output, but still run `ChannelSimulator` + `BS` + `UE` because the BS transmits and the UE receives the communication stream.

There are two ways to disable sensing RX: set `enable_sensing_rx: false`, or set
`sensing_rx_channel_count: 0`.

To view monostatic sensing results, run `python3 scripts/plot_sensing_fast.py`. As in
hardware mode, RD streaming starts when the viewer connects.

## Configuring the channel (`simulation:` block in BS_Sim.yaml)

```yaml
radio_backend: sim
simulation:
  session: oisac_sim             # must match across all three processes
  enable_comm_rx: true           # produce the comm RX path (false = sensing-only, no UE)
  enable_sensing_rx: true        # produce the sensing RX paths (false = comm-only)
  noise_power_dbfs: -50          # AWGN per RX channel; <= -200 disables
  cfo_hz: 0.0                    # relative CFO on the communication/bistatic RX
  timing_offset_samples: 0       # fixed integer sample offset, folded into path delays
  array_spacing_m: 0.04283       # physical ULA spacing (m); d/λ scales with center_freq
                                 # (42.83 mm = λ/2 @ 3.5 GHz, matches the sensing viewers).
                                 # Set <= 0 to fall back to the legacy array_spacing_lambda.
  array_spacing_lambda: 0.5      # legacy fixed spacing (wavelengths); used only if array_spacing_m <= 0
  ring_capacity_samples: 262144  # ~2 frames; small keeps TX close to the hub
  steering_override_file: ""     # array manifold: [num_targets x num_channels] complex<float>, row-major; empty = ULA
  comm_multipath_taps:           # communication LoS path + static multipath paths
    - { delay_samples: 0, gain_db: 0, phase_deg: 0 }
  targets:                       # monostatic sensing scatterers (with array response)
    - { range_m: 30, velocity_mps: 5,  gain_db: -6,  angle_deg: 20 }
    - { range_m: 75, velocity_mps: -3, gain_db: -12, angle_deg: -10 }
  # bistatic_targets:            # optional independent scene for the bistatic (comm) channel
  #   - { range_m: 45, velocity_mps: 8, gain_db: -8, angle_deg: 0 }
```

The number of sensing antennas equals the number of `sensing_rx_channels` entries.

`targets` describes the scatterer scene seen by monostatic sensing and generates the
corresponding array response for each sensing antenna. The bistatic (communication)
channel can be configured separately with `bistatic_targets`; if it is omitted, the
simulator reuses `targets`, so the same scatterer scene drives both the monostatic
and bistatic chains. Because the communication RX is single-antenna, `angle_deg` is
not used in the bistatic channel calculation. To override the array manifold, provide
`num_targets × num_channels` little-endian `complex<float>` values in row-major order
via `steering_override_file`; each row corresponds to one target.

## Channel model

The model is applied to transmitted samples \(x[n]\) in this order.

1. For each scatterer, compute integer propagation delay, Doppler, and complex gain; the monostatic sensing path also applies the array manifold vector.
2. The communication/bistatic path first adds the LoS path and static multipath paths, then adds the scatterer-return components.
3. The communication/bistatic path applies relative CFO to the whole received signal; the monostatic sensing path does not apply CFO.
4. AWGN is added to every RX output.

Target round-trip delay, sample delay, and Doppler are:

$$
\tau_i = \frac{2 R_i}{c}, \qquad
\ell_i = \operatorname{round}(\tau_i f_s) + n_0, \qquad
f_{D,i} = \frac{2 v_i}{\lambda}, \qquad
\lambda = \frac{c}{f_c}
$$

Here \(R_i\) is `range_m`, \(v_i\) is `velocity_mps`, \(f_s\) is `sample_rate`,
\(f_c\) is `center_freq`, and \(n_0\) is `timing_offset_samples`. The \(n_0\)
term is a fixed integer sample offset; the simulator does not model sampling
frequency offset (SFO).

The ULA array manifold is:

$$
a_{i,k}(\theta_i) =
\exp\!\left(j 2\pi \frac{d}{\lambda} k \sin\theta_i\right)
$$

where \(k\) is the antenna index. The electrical spacing is:

$$
\frac{d}{\lambda} = \frac{\texttt{array\_spacing\_m}\, f_c}{c}
$$

It is derived from the physical spacing and carrier frequency, so the recovered angle
remains correct at any `center_freq`; the viewers invert the phase slope using the
same physical spacing. With `array_spacing_m <= 0`, the frequency-independent legacy
parameter `array_spacing_lambda` is used instead. If `steering_override_file` is set,
the simulator reads \(a_{i,k}\) directly from the array manifold matrix.

For monostatic sensing RX antenna \(k\):

$$
y_{\mathrm{mono},k}[n]
= \sum_i g_i\, a_{i,k}(\theta_i)\, x[n-\ell_i]\,
  e^{j 2\pi f_{D,i} n / f_s}
  + w_k[n]
$$

The monostatic sensing channel shares the simulator clock with the transmitter and
does not apply relative CFO. The simulator does not model sampling frequency offset
(SFO). In the equation above, \(w_k[n]\) is AWGN.

The communication/bistatic RX is single-antenna. The LoS/static multipath paths first
form the communication multipath component:

$$
u_{\mathrm{LoS}}[n] =
\sum_p h_p\, x[n-\ell_p],
\qquad
\ell_p = d_p + n_0
$$

Here \(d_p\) comes from `comm_multipath_taps[].delay_samples`, and \(h_p\) is determined
by `gain_db` and `phase_deg`. Scatterer-return components are added to the same
communication channel:

$$
u[n] =
u_{\mathrm{LoS}}[n]
+ \sum_i g_i\, x[n-\ell_i]\, e^{j 2\pi f_{D,i} n / f_s}
$$

The communication/bistatic chain then applies relative CFO and AWGN:

$$
y_{\mathrm{comm}}[n] =
u[n]\, e^{j 2\pi f_{\mathrm{CFO}} n / f_s}
+ w_{\mathrm{comm}}[n]
$$

By default, `targets` drives both the monostatic sensing antennas (with array manifold)
and the bistatic communication channel (single-antenna, no array manifold), modelling
one coherent scene. Set `bistatic_targets` to give the bistatic/communication channel
its own independent scatterers. `comm_multipath_taps` configures the delay, gain, and
initial phase of the LoS path and static multipath paths.

## Notes / limits

- Each process stops cleanly on Ctrl-C (SIGINT), independently of the others; the hub
  unlinks its shared memory on exit.
- Not real-time: pacing is by shared-memory backpressure (correctness over speed).
- Keep `ring_capacity_samples` small (≈ a couple of frames) so the transmitter cannot
  race far ahead of the hub and overflow the monostatic TX/RX pairing queue.
- The simulator does not model sampling frequency offset (SFO); `timing_offset_samples`
  is only a fixed integer sample offset.
- The simulator does not model fractional delay; all propagation delays and configured
  timing offsets are quantized to integer samples.
