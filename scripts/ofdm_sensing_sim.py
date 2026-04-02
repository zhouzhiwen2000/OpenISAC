#!/usr/bin/env python3
import argparse
import sys
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def parse_scalar(text):
    text = text.strip()
    if not text:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    # Try int/float
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except Exception:
        return text


def parse_simple_yaml(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Strip comments
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if not val:
                continue
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                if not inner:
                    data[key] = []
                    continue
                items = [v.strip() for v in inner.split(",")]
                data[key] = [parse_scalar(v) for v in items if v]
            else:
                data[key] = parse_scalar(val)
    return data


def parse_targets(text):
    # Format: "range_m,vel_mps,amp,angle_deg;range_m,vel_mps,amp,angle_deg"
    if not text:
        return None
    targets = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = [p.strip() for p in part.split(",")]
        if len(fields) < 2:
            raise ValueError("Each target needs at least range_m and vel_mps")
        r = float(fields[0])
        v = float(fields[1])
        a = float(fields[2]) if len(fields) >= 3 else 1.0
        ang = float(fields[3]) if len(fields) >= 4 else 0.0
        targets.append((r, v, a, ang))
    return targets


def default_config_path():
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "config" / "Demodulator_X310.yaml",
        repo_root / "config" / "Modulator_X310.yaml",
        repo_root / "config" / "Demodulator_B210.yaml",
        repo_root / "config" / "Modulator_B210.yaml",
        Path("config") / "Demodulator_X310.yaml",
        Path("config") / "Modulator_X310.yaml",
        Path("config") / "Demodulator_B210.yaml",
        Path("config") / "Modulator_B210.yaml",
        Path("Demodulator_X310.yaml"),
        Path("Modulator_X310.yaml"),
        Path("Demodulator_B210.yaml"),
        Path("Modulator_B210.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def qpsk_symbols(rng, shape):
    bits = rng.integers(0, 4, size=shape)
    mapping = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / math.sqrt(2)
    return mapping[bits]


def steering_vector(num_antennas, angle_deg, spacing_lambda):
    theta = np.deg2rad(angle_deg)
    m = np.arange(num_antennas, dtype=np.float64)
    return np.exp(1j * 2.0 * np.pi * spacing_lambda * m * np.sin(theta))


def build_frame(rng, fft_size, cp_length, sample_rate, center_freq, num_symbols,
                pilot_positions, targets, frame_idx, strd, num_antennas, spacing_lambda):
    c = 299792458.0
    lam = c / center_freq

    delta_f = sample_rate / fft_size
    k = np.arange(fft_size)
    k_shift = np.where(k < fft_size // 2, k, k - fft_size)
    subcarrier_freq = k_shift * delta_f

    t_sym = (fft_size + cp_length) / sample_rate
    m = np.arange(num_symbols)
    # STRD: sensing symbols are spaced by strd * t_sym (not continuous)
    t_m = (frame_idx * num_symbols + m) * t_sym * strd

    # Transmit symbols in frequency domain (unshifted indexing)
    X = qpsk_symbols(rng, (num_symbols, fft_size)).astype(np.complex64)
    # Set DC to zero
    X[:, 0] = 0.0
    # Insert pilots as 1+0j
    for p in pilot_positions:
        try:
            idx = int(p)
            if 0 <= idx < fft_size:
                X[:, idx] = 1.0 + 0.0j
        except Exception:
            pass

    # Channel response: (M, num_symbols, fft_size)
    H = np.zeros((num_antennas, num_symbols, fft_size), dtype=np.complex128)
    for r_m, v_mps, amp, ang in targets:
        tau = 2.0 * r_m / c
        fd = 2.0 * v_mps / lam
        phase_delay = np.exp(-1j * 2.0 * np.pi * subcarrier_freq * tau)
        phase_dopp = np.exp(1j * 2.0 * np.pi * fd * t_m)
        a_theta = steering_vector(num_antennas, ang, spacing_lambda)
        contrib = amp * phase_dopp[:, None] * phase_delay[None, :]
        H += a_theta[:, None, None] * contrib[None, :, :]

    # Received symbols
    Y = X[None, :, :] * H
    return X, Y, t_sym, subcarrier_freq


def add_awgn(rng, Y, snr_db):
    sig_power = np.mean(np.abs(Y) ** 2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise = (np.sqrt(noise_power / 2.0) *
             (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape)))
    return Y + noise


def main():
    parser = argparse.ArgumentParser(description="OFDM sensing simulation using system YAML parameters")
    parser.add_argument("--config", type=str, default=None, help="Path to Modulator/Demodulator YAML")
    parser.add_argument("--snr", type=float, default=30.0, help="SNR in dB")
    parser.add_argument("--targets", type=str, default=None,
                        help='Targets: "range_m,vel_mps,amp,angle_deg;..."')
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--frames", type=int, default=1, help="Number of frames to simulate")
    parser.add_argument("--strd", type=int, default=20, help="Sensing symbol stride (STRD)")
    parser.add_argument("--rx_ant", type=int, default=1, help="Number of sensing RX antennas (M)")
    parser.add_argument("--array_spacing", type=float, default=0.5,
                        help="Inter-element spacing in wavelengths (d/lambda)")
    parser.add_argument("--look_angle", type=float, default=0.0,
                        help="Beamforming look angle in degrees (used when M>1)")
    parser.add_argument("--angle_min", type=float, default=-60.0, help="Angle scan min (deg)")
    parser.add_argument("--angle_max", type=float, default=60.0, help="Angle scan max (deg)")
    parser.add_argument("--angle_bins", type=int, default=121, help="Angle scan bins")
    parser.add_argument("--delay_offset", type=int, default=2, help="Manual delay bin offset")
    parser.add_argument("--doppler_offset", type=int, default=2, help="Manual Doppler bin offset")
    parser.add_argument("--display_range_window", type=float, default=80.0,
                        help="Display range window beyond first target (m), starting from 0")
    parser.add_argument("--display_vel_window", type=float, default=12.0,
                        help="Display velocity window around target (m/s), symmetric")
    parser.add_argument("--plot", action="store_true", help="Show plots (no files saved)")
    parser.add_argument("--plot_ad_phase", action="store_true",
                        help="Show per-antenna angle-Doppler phase maps (off by default)")
    parser.add_argument("--unwrap_phase", action="store_true",
                        help="Unwrap phase across antennas for phase-antenna plot")
    args = parser.parse_args()

    cfg_path = args.config or default_config_path()
    if not cfg_path:
        print("No YAML config found. Use --config to specify one or place templates under config/.")
        sys.exit(1)
    cfg = parse_simple_yaml(cfg_path)

    fft_size = int(cfg.get("fft_size", 1024))
    cp_length = int(cfg.get("cp_length", 128))
    sample_rate = float(cfg.get("sample_rate", 50e6))
    center_freq = float(cfg.get("center_freq", 2.4e9))
    num_symbols = int(cfg.get("sensing_symbol_num", cfg.get("num_symbols", 64)))
    range_fft_size = int(cfg.get("range_fft_size", fft_size)) * 10
    doppler_fft_size = int(cfg.get("doppler_fft_size", num_symbols)) * 10
    pilot_positions = cfg.get("pilot_positions", []) or []

    rng = np.random.default_rng(args.seed)

    # Targets
    targets = parse_targets(args.targets)
    if targets is None:
        # Default example target: single target with larger range, Doppler, and angle
        targets = [(50.0, 5.0, 1.0, 20.0)]

    c = 299792458.0
    lam = c / center_freq
    delta_f = sample_rate / fft_size

    total_frames = max(1, args.frames)
    rd_mag = None
    peak_ranges = []
    peak_vels = []
    eps = 1e-12

    last_range_time = None
    last_rd_mag = None
    last_range_m = None
    last_vel_mps = None

    strd = max(1, int(args.strd))
    num_antennas = max(1, int(args.rx_ant))
    spacing_lambda = float(args.array_spacing)
    look_angle = float(args.look_angle)
    last_range_time_m = None
    last_rd_m = None
    last_range_bin_est = None
    last_doppler_bin_est = None
    last_range_bin_adj = None
    last_doppler_bin_adj = None

    for i in range(total_frames):
        X, Y, t_sym, subcarrier_freq = build_frame(
            rng, fft_size, cp_length, sample_rate, center_freq, num_symbols,
            pilot_positions, targets, frame_idx=i, strd=strd,
            num_antennas=num_antennas, spacing_lambda=spacing_lambda
        )
        Y_noisy = add_awgn(rng, Y, args.snr)

        # Channel estimation (avoid divide-by-zero warnings)
        H_est = np.zeros_like(Y_noisy)
        mask = np.abs(X) > eps
        H_est[:, mask] = Y_noisy[:, mask] / X[mask]

        # Beamform across antennas if M > 1 (for RD display/summary)
        if num_antennas > 1:
            a_look = steering_vector(num_antennas, look_angle, spacing_lambda)
            weights = np.conj(a_look) / num_antennas
            H_combined = np.tensordot(weights, H_est, axes=(0, 0))
        else:
            H_combined = H_est[0]

        # Optional windowing (combined)
        range_win = np.hanning(fft_size).astype(np.float64)
        doppler_win = np.hanning(num_symbols).astype(np.float64)
        H_win = H_combined * range_win[None, :]
        H_win = H_win * doppler_win[:, None]

        # Range FFT (IFFT across subcarriers)
        range_time = np.fft.ifft(H_win, n=range_fft_size, axis=1)
        # Doppler FFT across symbols
        rd = np.fft.fftshift(np.fft.fft(range_time, n=doppler_fft_size, axis=0), axes=0)
        rd_mag = 20.0 * np.log10(np.abs(rd) + 1e-12)

        # Per-antenna processing for angle/Doppler phase plots
        H_win_m = H_est * range_win[None, None, :]
        H_win_m = H_win_m * doppler_win[None, :, None]
        range_time_m = np.fft.ifft(H_win_m, n=range_fft_size, axis=2)
        rd_m = np.fft.fftshift(np.fft.fft(range_time_m, n=doppler_fft_size, axis=1), axes=1)

        # Quick peak summary
        peak_idx = np.unravel_index(np.argmax(rd_mag), rd_mag.shape)
        peak_vel_idx, peak_range_idx = peak_idx
        delay = np.arange(range_fft_size) / (range_fft_size * delta_f)
        range_m = delay * c / 2.0
        doppler_hz = np.fft.fftshift(np.fft.fftfreq(doppler_fft_size, d=t_sym * strd))
        vel_mps = doppler_hz * lam / 2.0
        peak_ranges.append(float(range_m[peak_range_idx]))
        peak_vels.append(float(vel_mps[peak_vel_idx]))

        last_range_time = range_time
        last_rd_mag = rd_mag
        last_range_m = range_m
        last_vel_mps = vel_mps
        last_range_time_m = range_time_m
        last_rd_m = rd_m
        last_range_bin_est = int(peak_range_idx)
        last_doppler_bin_est = int(peak_vel_idx)
        last_range_bin_adj = int(np.clip(peak_range_idx + args.delay_offset, 0, range_fft_size - 1))
        last_doppler_bin_adj = int(np.clip(peak_vel_idx + args.doppler_offset, 0, doppler_fft_size - 1))

    print("OFDM sensing simulation complete")
    print(f"Config: {cfg_path}")
    print(f"FFT: {fft_size}, CP: {cp_length}, fs: {sample_rate:.3f} Hz")
    print(f"Symbols: {num_symbols}, Range FFT: {range_fft_size}, Doppler FFT: {doppler_fft_size}")
    print(f"STRD: {strd} (sensing symbol stride)")
    print(f"RX Antennas: {num_antennas}, Spacing: {spacing_lambda} lambda, Look angle: {look_angle} deg")
    print(f"Targets: {targets}")
    if peak_ranges:
        avg_range = sum(peak_ranges) / len(peak_ranges)
        avg_vel = sum(peak_vels) / len(peak_vels)
        print(f"Frames: {total_frames}, Peak range avg: {avg_range:.3f} m, Peak vel avg: {avg_vel:.3f} m/s")
        if last_range_bin_est is not None:
            est_r = last_range_m[last_range_bin_est]
            est_v = last_vel_mps[last_doppler_bin_est]
            adj_r = last_range_m[last_range_bin_adj]
            adj_v = last_vel_mps[last_doppler_bin_adj]
            print(f"Est bin (r,d): ({last_range_bin_est}, {last_doppler_bin_est}) -> ({est_r:.3f} m, {est_v:.3f} m/s)")
            print(f"Adj bin (r,d): ({last_range_bin_adj}, {last_doppler_bin_adj}) -> ({adj_r:.3f} m, {adj_v:.3f} m/s)")

    if args.plot and last_rd_mag is not None:
        if targets:
            r_center = float(targets[0][0])
            v_center = float(targets[0][1])
        else:
            r_center = float(last_range_m[last_range_bin_adj]) if last_range_bin_adj is not None else 0.0
            v_center = float(last_vel_mps[last_doppler_bin_adj]) if last_doppler_bin_adj is not None else 0.0
        r_win = max(0.0, float(args.display_range_window))
        v_win = max(0.0, float(args.display_vel_window))
        r_min = 0.0
        r_max = r_center + r_win if r_win > 0 else last_range_m[-1]
        v_max = abs(v_center) + v_win if v_win > 0 else max(abs(last_vel_mps[0]), abs(last_vel_mps[-1]))
        range_mask = (last_range_m >= r_min) & (last_range_m <= r_max)
        vel_mask = (last_vel_mps >= -v_max) & (last_vel_mps <= v_max)
        if not np.any(range_mask):
            range_mask = np.ones_like(last_range_m, dtype=bool)
        if not np.any(vel_mask):
            vel_mask = np.ones_like(last_vel_mps, dtype=bool)

        fig_rd = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        rd_disp = last_rd_mag[np.ix_(vel_mask, range_mask)]
        range_disp = last_range_m[range_mask]
        vel_disp = last_vel_mps[vel_mask]
        extent = [range_disp[0], range_disp[-1], vel_disp[0], vel_disp[-1]]
        im = ax.imshow(rd_disp, aspect="auto", origin="lower", extent=extent, cmap="viridis")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Range-Doppler Map")
        if last_range_bin_adj is not None and last_doppler_bin_adj is not None:
            sel_r = last_range_m[last_range_bin_adj]
            sel_v = last_vel_mps[last_doppler_bin_adj]
            ax.plot([sel_r], [sel_v], marker="o", markersize=6, color="white", markeredgecolor="black")
            ax.annotate(
                f"bin(r,d)=({last_range_bin_adj},{last_doppler_bin_adj})",
                xy=(sel_r, sel_v),
                xytext=(10, 10),
                textcoords="offset points",
                color="white",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="white", alpha=0.6),
            )
        plt.colorbar(im, ax=ax, label="Magnitude (dB)")
        plt.tight_layout()

        plt.figure(figsize=(10, 4))
        rp = np.mean(np.abs(last_range_time), axis=0)
        plt.plot(range_disp, 20.0 * np.log10(rp[range_mask] + 1e-12))
        plt.xlabel("Range (m)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Range Profile (mean over Doppler)")
        plt.tight_layout()

        # Angle-Doppler phase per antenna at adjusted range bin (optional)
        if args.plot_ad_phase and last_rd_m is not None and last_range_bin_adj is not None:
            angles = np.linspace(args.angle_min, args.angle_max, max(2, args.angle_bins))
            steer = np.zeros((angles.size, num_antennas), dtype=np.complex128)
            for ai, ang in enumerate(angles):
                steer[ai, :] = steering_vector(num_antennas, ang, spacing_lambda)

            ncols = int(np.ceil(np.sqrt(num_antennas)))
            nrows = int(np.ceil(num_antennas / ncols))
            fig_ad, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
            for m in range(num_antennas):
                phase_map = np.angle(np.conj(steer[:, m])[:, None] *
                                     last_rd_m[m, :, last_range_bin_adj][None, :])
                axm = axes[m // ncols][m % ncols]
                extent_ad = [angles[0], angles[-1], last_vel_mps[0], last_vel_mps[-1]]
                im_ad = axm.imshow(phase_map.T, aspect="auto", origin="lower", extent=extent_ad,
                                   cmap="twilight")
                axm.set_title(f"Antenna {m} phase")
                axm.set_xlabel("Angle (deg)")
                axm.set_ylabel("Velocity (m/s)")
            for k in range(num_antennas, nrows * ncols):
                axes[k // ncols][k % ncols].axis("off")
            fig_ad.suptitle("Per-antenna Angle-Doppler Phase (at adjusted range bin)")
            fig_ad.colorbar(im_ad, ax=axes, shrink=0.8, label="Phase (rad)")
            fig_ad.tight_layout()

        # Phase vs antenna index: click RD map to update
        if last_rd_m is not None and num_antennas >= 1:
            if len(targets) > 0:
                ref_angle = float(targets[0][3]) if len(targets[0]) >= 4 else look_angle
            else:
                ref_angle = look_angle
            ref_phi = np.angle(steering_vector(num_antennas, ref_angle, spacing_lambda))
            if args.unwrap_phase and num_antennas > 1:
                ref_phi = np.unwrap(ref_phi)

            fig_pa, ax_phase = plt.subplots(figsize=(7, 4))
            ant_idx = np.arange(num_antennas)
            line_phase, = ax_phase.plot([], [], marker="o", label="Measured phase")
            line_ref, = ax_phase.plot(ant_idx, ref_phi, marker="x", linestyle="--", label="Steering phase")
            ax_phase.set_xlabel("Antenna Index")
            ax_phase.set_ylabel("Phase (rad, unwrapped)" if args.unwrap_phase else "Phase (rad)")
            ax_phase.set_title("Phase vs Antenna Index (click RD map)")
            ax_phase.grid(True, alpha=0.3)
            ax_phase.legend(loc="upper left")

            ax_amp = ax_phase.twinx()
            line_amp, = ax_amp.plot([], [], color="tab:orange", marker="s", linestyle="-",
                                    label="Measured magnitude")
            ax_amp.set_ylabel("Magnitude")
            ax_amp.legend(loc="upper right")

            def _update_phase_plot(r_idx, d_idx):
                phi = np.angle(last_rd_m[:, d_idx, r_idx])
                amp = np.abs(last_rd_m[:, d_idx, r_idx])
                if args.unwrap_phase and num_antennas > 1:
                    phi_u = np.unwrap(phi)
                else:
                    phi_u = phi
                line_phase.set_data(ant_idx, phi_u)
                line_amp.set_data(ant_idx, amp)
                ax_phase.relim()
                ax_phase.autoscale_view()
                ax_amp.relim()
                ax_amp.autoscale_view()
                r_val = last_range_m[r_idx]
                v_val = last_vel_mps[d_idx]
                ax_phase.set_title(f"Phase vs Antenna Index (r={r_val:.3f} m, v={v_val:.3f} m/s)")
                fig_pa.canvas.draw_idle()

            if last_range_bin_adj is not None and last_doppler_bin_adj is not None:
                _update_phase_plot(last_range_bin_adj, last_doppler_bin_adj)

            def _on_click(event):
                if event.inaxes != ax:
                    return
                if event.xdata is None or event.ydata is None:
                    return
                r_idx = int(np.argmin(np.abs(last_range_m - event.xdata)))
                d_idx = int(np.argmin(np.abs(last_vel_mps - event.ydata)))
                _update_phase_plot(r_idx, d_idx)

            fig_rd.canvas.mpl_connect("button_press_event", _on_click)

            # Phase slope vs antenna index (2D over range & Doppler)
            if num_antennas >= 2:
                phi_m = np.unwrap(np.angle(last_rd_m), axis=0)
                m_idx = np.arange(num_antennas, dtype=np.float64)
                m_centered = m_idx - np.mean(m_idx)
                denom = np.sum(m_centered ** 2)
                slope = np.sum(m_centered[:, None, None] * phi_m, axis=0) / denom

                plt.figure(figsize=(10, 6))
                ax_pd = plt.gca()
                slope_disp = slope[np.ix_(vel_mask, range_mask)]
                extent_pd = [range_disp[0], range_disp[-1], vel_disp[0], vel_disp[-1]]
                im_pd = ax_pd.imshow(slope_disp, aspect="auto", origin="lower",
                                     extent=extent_pd, cmap="coolwarm", alpha=1.0)
                ax_pd.set_xlabel("Range (m)")
                ax_pd.set_ylabel("Velocity (m/s)")
                ax_pd.set_title("Phase Slope vs Antenna Index")
                plt.colorbar(im_pd, ax=ax_pd,
                             label="Phase slope (rad/index)" + (" (unwrapped)" if args.unwrap_phase else ""))

                # Phase difference between antenna 2 and 1 (2D over range & Doppler)
                phi_diff = np.angle(last_rd_m[1, :, :] * np.conj(last_rd_m[0, :, :]))
                phi_diff_disp = phi_diff[np.ix_(vel_mask, range_mask)]
                plt.figure(figsize=(10, 6))
                ax_pd2 = plt.gca()
                im_pd2 = ax_pd2.imshow(phi_diff_disp, aspect="auto", origin="lower",
                                       extent=extent_pd, cmap="twilight", alpha=1.0)
                ax_pd2.set_xlabel("Range (m)")
                ax_pd2.set_ylabel("Velocity (m/s)")
                ax_pd2.set_title("Phase Difference: Antenna 2 - Antenna 1")
                plt.colorbar(im_pd2, ax=ax_pd2,
                             label="Phase diff (rad, unwrapped)" if args.unwrap_phase else "Phase diff (rad)")
        plt.show()


if __name__ == "__main__":
    main()
