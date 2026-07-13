#!/usr/bin/env python3
"""Compare eRTM correlation peak estimators with an offline Monte-Carlo simulation.

The simulator mirrors the CPU UE eRTM timing-offset estimator:
1. Generate two frequency-domain channel estimates with a known relative delay.
2. Rebuild the oversampled delay spectra using the same bin placement as UE.cpp.
3. Correlate either delay magnitudes or complex delay profiles.
4. Compare integer argmax against fractional peak refinements.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SNR_DB = "0,5,10,15,20,30"
DEFAULT_TRUE_SHIFTS = "-0.45,-0.35,-0.25,-0.15,-0.05,0,0.05,0.15,0.25,0.35,0.45"
ESTIMATORS = ("integer", "parabolic", "centroid3", "centroid5", "gaussian", "quinn_complex")


@dataclass
class Estimate:
    shift_bins: float
    shift_samples: float
    peak_index: int
    metric: float


@dataclass
class SummaryRow:
    snr_db: float
    true_shift_samples: str
    estimator: str
    trials: int
    bias_samples: float
    rmse_samples: float
    std_samples: float
    p95_abs_error_samples: float
    outlier_rate: float
    mean_metric: float


@dataclass
class ExamplePeak:
    snr_db: float
    true_shift_samples: float
    corr: np.ndarray
    estimates: dict[str, Estimate]


def parse_float_list(text: str) -> list[float]:
    """Parse comma lists or inclusive start:stop:step ranges."""
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [float(part) for part in item.split(":")]
            if len(parts) != 3:
                raise argparse.ArgumentTypeError(f"Range '{item}' must be start:stop:step")
            start, stop, step = parts
            if step == 0.0:
                raise argparse.ArgumentTypeError("Range step must be non-zero")
            current = start
            # Include the endpoint within a small tolerance so 0:1:0.1 is intuitive.
            if step > 0.0:
                while current <= stop + abs(step) * 1e-9:
                    values.append(current)
                    current += step
            else:
                while current >= stop - abs(step) * 1e-9:
                    values.append(current)
                    current += step
        else:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required")
    return values


def parse_estimators(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("At least one estimator is required")
    unknown = sorted(set(names) - set(ESTIMATORS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown estimator(s): {', '.join(unknown)}; choices: {', '.join(ESTIMATORS)}"
        )
    return names


def normalize_negative_option_value(argv: list[str], option: str) -> list[str]:
    """Allow argparse options such as ``--true-shifts -0.3,0,0.3``."""
    normalized = list(argv)
    for idx, token in enumerate(normalized[:-1]):
        if token == option and normalized[idx + 1].startswith("-"):
            normalized[idx] = f"{option}={normalized[idx + 1]}"
            del normalized[idx + 1]
            break
    return normalized


def parse_args() -> argparse.Namespace:
    argv = normalize_negative_option_value(sys.argv[1:], "--true-shifts")
    argv = normalize_negative_option_value(argv, "--snr-db")
    parser = argparse.ArgumentParser(
        description="Compare eRTM TO correlation peak estimators on synthetic channels."
    )
    parser.add_argument("--fft-size", type=int, default=1024)
    parser.add_argument("--oversample", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--snr-db", type=parse_float_list, default=parse_float_list(DEFAULT_SNR_DB))
    parser.add_argument("--true-shifts", type=parse_float_list, default=parse_float_list(DEFAULT_TRUE_SHIFTS))
    parser.add_argument("--taps", type=int, default=6)
    parser.add_argument("--delay-spread-samples", type=float, default=64.0)
    parser.add_argument(
        "--profile",
        choices=("magnitude", "complex"),
        default="magnitude",
        help="Delay profile used for the main estimators; quinn_complex always uses complex correlation.",
    )
    parser.add_argument("--estimators", type=parse_estimators, default=list(ESTIMATORS))
    parser.add_argument("--outlier-threshold-samples", type=float, default=0.25)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV path for per-shift statistics.")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib comparison plots.")
    parser.add_argument("--plot-output", type=Path, default=None, help="Optional path to save plots instead of showing them.")
    parser.add_argument(
        "--print-detail",
        action="store_true",
        help="Print one row per SNR/true-shift/estimator instead of compact per-SNR summaries.",
    )
    return parser.parse_args(argv)


def signed_subcarrier_indices(fft_size: int) -> np.ndarray:
    half = fft_size // 2
    return np.concatenate(
        (np.arange(0, half, dtype=np.float64), np.arange(-half, 0, dtype=np.float64))
    )


def signed_bin_from_index(index: int, size: int) -> int:
    return int(index) if index < (size + 1) // 2 else int(index) - int(size)


def wrap_error_samples(error: np.ndarray | float, period_samples: float) -> np.ndarray | float:
    return (np.asarray(error) + period_samples / 2.0) % period_samples - period_samples / 2.0


def generate_sparse_channel_freq(
    fft_size: int,
    taps: int,
    delay_spread_samples: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one FFTW-native ordered channel response with fractional sparse taps."""
    if fft_size <= 0 or fft_size % 2 != 0:
        raise ValueError("fft_size must be a positive even integer")
    if taps <= 0:
        raise ValueError("taps must be positive")
    if delay_spread_samples <= 0.0:
        raise ValueError("delay_spread_samples must be positive")

    delays = np.sort(rng.uniform(0.0, delay_spread_samples, size=taps))
    delays[0] = 0.0
    decay = np.exp(-delays / max(delay_spread_samples / 2.5, 1e-9))
    random_phasors = rng.normal(size=taps) + 1j * rng.normal(size=taps)
    amplitudes = random_phasors * decay / math.sqrt(2.0)
    amplitudes[0] += 1.0 + 0.0j

    k = signed_subcarrier_indices(fft_size)
    phase = np.exp(-1j * 2.0 * math.pi * np.outer(k, delays) / float(fft_size))
    channel = phase @ amplitudes
    rms = math.sqrt(float(np.mean(np.abs(channel) ** 2)))
    if rms > 0.0:
        channel = channel / rms
    return channel.astype(np.complex128)


def apply_fractional_delay(channel_freq: np.ndarray, shift_samples: float) -> np.ndarray:
    fft_size = channel_freq.size
    k = signed_subcarrier_indices(fft_size)
    phase = np.exp(-1j * 2.0 * math.pi * k * float(shift_samples) / float(fft_size))
    return channel_freq * phase


def add_channel_awgn(
    channel_freq: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    signal_power = float(np.mean(np.abs(channel_freq) ** 2))
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    sigma = math.sqrt(noise_power / 2.0)
    noise = rng.normal(0.0, sigma, size=channel_freq.shape) + 1j * rng.normal(
        0.0, sigma, size=channel_freq.shape
    )
    return channel_freq + noise


def compute_oversampled_delay_spectrum(
    channel_freq: np.ndarray,
    oversample: int,
    shift_samples: float = 0.0,
) -> np.ndarray:
    """NumPy equivalent of UE.cpp::_compute_ertm_oversampled_delay_spectrum()."""
    fft_size = channel_freq.size
    os_size = fft_size * oversample
    half = fft_size // 2
    if fft_size <= 0 or oversample <= 0 or fft_size % 2 != 0:
        raise ValueError("fft_size must be even and oversample must be positive")

    os_in = np.zeros(os_size, dtype=np.complex128)
    apply_shift = math.isfinite(shift_samples) and shift_samples != 0.0
    # Runtime H_est is FFTW-native: [0, ..., N/2-1, -N/2, ..., -1].
    # The eRTM delay IFFT works on natural order followed by zero padding.
    neg = channel_freq[half:].copy()
    pos = channel_freq[:half].copy()
    if apply_shift:
        neg_k = np.arange(-half, 0, dtype=np.float64)
        pos_k = np.arange(0, half, dtype=np.float64)
        pos *= np.exp(-1j * 2.0 * math.pi * pos_k * shift_samples / float(fft_size))
        neg *= np.exp(-1j * 2.0 * math.pi * neg_k * shift_samples / float(fft_size))
    os_in[:half] = neg
    os_in[half:fft_size] = pos

    # FFTW_BACKWARD is unnormalized; numpy.ifft includes 1/os_size.
    return np.fft.ifft(os_in) * float(os_size) / math.sqrt(float(fft_size))


def correlate_delay_magnitudes(
    bs_uplink_delay: np.ndarray,
    ue_downlink_delay: np.ndarray,
) -> np.ndarray:
    bs_mag = np.abs(bs_uplink_delay)
    ue_mag = np.abs(ue_downlink_delay)
    denom = math.sqrt(float(np.sum(bs_mag * bs_mag)) * float(np.sum(ue_mag * ue_mag)))
    if not math.isfinite(denom) or denom <= 0.0:
        raise RuntimeError("Cannot normalize zero-energy delay spectra")
    corr = np.fft.ifft(np.conj(np.fft.fft(bs_mag)) * np.fft.fft(ue_mag)).real
    return corr / denom


def correlate_delay_complex(
    bs_uplink_delay: np.ndarray,
    ue_downlink_delay: np.ndarray,
) -> np.ndarray:
    """Phase-aware complex circular correlation of two delay spectra."""
    denom = math.sqrt(
        float(np.sum(np.abs(bs_uplink_delay) ** 2)) *
        float(np.sum(np.abs(ue_downlink_delay) ** 2))
    )
    if not math.isfinite(denom) or denom <= 0.0:
        raise RuntimeError("Cannot normalize zero-energy delay spectra")
    corr = np.fft.ifft(np.conj(np.fft.fft(bs_uplink_delay)) * np.fft.fft(ue_downlink_delay))
    return corr / denom


def profile_correlation(
    bs_uplink_delay: np.ndarray,
    ue_downlink_delay: np.ndarray,
    profile: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    magnitude_corr = correlate_delay_magnitudes(bs_uplink_delay, ue_downlink_delay)
    complex_corr = correlate_delay_complex(bs_uplink_delay, ue_downlink_delay)
    if profile == "magnitude":
        return magnitude_corr, complex_corr
    if profile == "complex":
        return np.abs(complex_corr), complex_corr
    raise ValueError(f"Unknown profile: {profile}")


def argmax_peak(corr: np.ndarray) -> tuple[int, int, float]:
    peak_index = int(np.argmax(corr))
    signed_peak = signed_bin_from_index(peak_index, corr.size)
    return peak_index, signed_peak, float(corr[peak_index])


def parabolic_delta(corr: np.ndarray, peak_index: int) -> float:
    n = corr.size
    y_left = float(corr[(peak_index - 1) % n])
    y_center = float(corr[peak_index])
    y_right = float(corr[(peak_index + 1) % n])
    curvature = y_left - 2.0 * y_center + y_right
    if not math.isfinite(curvature) or curvature >= -1e-15:
        return 0.0
    delta = 0.5 * (y_left - y_right) / curvature
    if not math.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -0.5, 0.5))


def gaussian_delta(corr: np.ndarray, peak_index: int) -> float:
    n = corr.size
    y_left = float(corr[(peak_index - 1) % n])
    y_center = float(corr[peak_index])
    y_right = float(corr[(peak_index + 1) % n])
    if y_left <= 0.0 or y_center <= 0.0 or y_right <= 0.0:
        return parabolic_delta(corr, peak_index)
    log_left = math.log(y_left)
    log_center = math.log(y_center)
    log_right = math.log(y_right)
    curvature = log_left - 2.0 * log_center + log_right
    if not math.isfinite(curvature) or curvature >= -1e-15:
        return parabolic_delta(corr, peak_index)
    delta = 0.5 * (log_left - log_right) / curvature
    if not math.isfinite(delta):
        return parabolic_delta(corr, peak_index)
    return float(np.clip(delta, -0.5, 0.5))


def centroid_delta(corr: np.ndarray, peak_index: int, radius: int) -> float:
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    indices = (peak_index + offsets.astype(np.int64)) % corr.size
    values = corr[indices].astype(np.float64, copy=False)
    weights = values - float(np.min(values))
    weights = np.maximum(weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.maximum(values, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return 0.0
    delta = float(np.sum(offsets * weights) / total)
    if not math.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -0.5, 0.5))


def quinn_complex_delta(spectrum: np.ndarray, peak_index: int) -> float:
    """Port of DelayProcessor::estimate_fractional_delay from OFDMCore.hpp."""
    n = spectrum.size
    if n < 3:
        return 0.0
    d0 = complex(spectrum[peak_index])
    d_prev = complex(spectrum[(peak_index - 1) % n])
    d_next = complex(spectrum[(peak_index + 1) % n])
    magnitude = abs(d0)
    epsilon = 1e-10
    if magnitude < epsilon:
        return 0.0

    denom = float(np.real(np.conj(d0) * d0))
    if denom > epsilon:
        alpha1 = float(np.real(np.conj(d_prev) * d0)) / denom
        alpha2 = float(np.real(np.conj(d_next) * d0)) / denom
    else:
        alpha1 = abs(d_prev) / (magnitude + epsilon)
        alpha2 = abs(d_next) / (magnitude + epsilon)

    alpha1 = float(np.clip(alpha1, -0.9999, 0.9999))
    alpha2 = float(np.clip(alpha2, -0.9999, 0.9999))
    delta1 = alpha1 / (1.0 - alpha1)
    delta2 = -alpha2 / (1.0 - alpha2)
    if not math.isfinite(delta1) or not math.isfinite(delta2):
        return 0.0

    abs1 = abs(delta1)
    abs2 = abs(delta2)
    if abs1 > 2.0 and abs2 > 2.0:
        delta = 0.5
    elif abs1 > 2.0:
        delta = delta2
    elif abs2 > 2.0:
        delta = delta1
    else:
        delta = delta2 if delta1 > 0.0 and delta2 > 0.0 else delta1
    return float(np.clip(delta, -0.5, 0.5))


def estimate_peak(
    corr: np.ndarray,
    oversample: int,
    estimator: str,
    complex_corr: np.ndarray | None = None,
) -> Estimate:
    peak_source = np.abs(complex_corr) if estimator == "quinn_complex" and complex_corr is not None else corr
    peak_index, signed_peak, metric = argmax_peak(peak_source)
    if estimator == "integer":
        delta = 0.0
    elif estimator == "parabolic":
        delta = parabolic_delta(corr, peak_index)
    elif estimator == "centroid3":
        delta = centroid_delta(corr, peak_index, radius=1)
    elif estimator == "centroid5":
        delta = centroid_delta(corr, peak_index, radius=2)
    elif estimator == "gaussian":
        delta = gaussian_delta(corr, peak_index)
    elif estimator == "quinn_complex":
        if complex_corr is None:
            raise ValueError("quinn_complex requires complex_corr")
        delta = quinn_complex_delta(complex_corr, peak_index)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    shift_bins = float(signed_peak) + delta
    return Estimate(
        shift_bins=shift_bins,
        shift_samples=shift_bins / float(oversample),
        peak_index=peak_index,
        metric=metric,
    )


def summarize_errors(
    snr_db: float,
    true_shift_samples: str,
    estimator: str,
    errors: Iterable[float],
    metrics: Iterable[float],
    outlier_threshold: float,
) -> SummaryRow:
    error_arr = np.asarray(list(errors), dtype=np.float64)
    metric_arr = np.asarray(list(metrics), dtype=np.float64)
    abs_error = np.abs(error_arr)
    return SummaryRow(
        snr_db=snr_db,
        true_shift_samples=true_shift_samples,
        estimator=estimator,
        trials=int(error_arr.size),
        bias_samples=float(np.mean(error_arr)),
        rmse_samples=float(math.sqrt(float(np.mean(error_arr * error_arr)))),
        std_samples=float(np.std(error_arr)),
        p95_abs_error_samples=float(np.percentile(abs_error, 95.0)),
        outlier_rate=float(np.mean(abs_error > outlier_threshold)),
        mean_metric=float(np.mean(metric_arr)),
    )


def run_simulation(args: argparse.Namespace) -> tuple[list[SummaryRow], list[SummaryRow], ExamplePeak]:
    if args.fft_size <= 0 or args.fft_size % 2 != 0:
        raise RuntimeError("--fft-size must be a positive even integer")
    if args.oversample <= 0:
        raise RuntimeError("--oversample must be positive")
    if args.trials <= 0:
        raise RuntimeError("--trials must be positive")

    rng = np.random.default_rng(args.seed)
    detail_errors: dict[tuple[float, float, str], list[float]] = defaultdict(list)
    detail_metrics: dict[tuple[float, float, str], list[float]] = defaultdict(list)
    aggregate_errors: dict[tuple[float, str], list[float]] = defaultdict(list)
    aggregate_metrics: dict[tuple[float, str], list[float]] = defaultdict(list)
    example: ExamplePeak | None = None

    for snr_db in args.snr_db:
        for true_shift in args.true_shifts:
            for _trial in range(args.trials):
                base = generate_sparse_channel_freq(
                    args.fft_size,
                    args.taps,
                    args.delay_spread_samples,
                    rng,
                )
                shifted = apply_fractional_delay(base, true_shift)
                noisy_bs = add_channel_awgn(base, snr_db, rng)
                noisy_ue = add_channel_awgn(shifted, snr_db, rng)
                bs_delay = compute_oversampled_delay_spectrum(noisy_bs, args.oversample)
                ue_delay = compute_oversampled_delay_spectrum(noisy_ue, args.oversample)
                corr, complex_corr = profile_correlation(bs_delay, ue_delay, args.profile)
                estimates = {
                    name: estimate_peak(corr, args.oversample, name, complex_corr)
                    for name in args.estimators
                }
                if example is None:
                    example = ExamplePeak(
                        snr_db=snr_db,
                        true_shift_samples=true_shift,
                        corr=corr.copy(),
                        estimates=estimates.copy(),
                    )
                for name, estimate in estimates.items():
                    error = float(wrap_error_samples(estimate.shift_samples - true_shift, args.fft_size))
                    detail_key = (snr_db, true_shift, name)
                    aggregate_key = (snr_db, name)
                    detail_errors[detail_key].append(error)
                    detail_metrics[detail_key].append(estimate.metric)
                    aggregate_errors[aggregate_key].append(error)
                    aggregate_metrics[aggregate_key].append(estimate.metric)

    if example is None:
        raise RuntimeError("No simulation data generated")

    detail_rows: list[SummaryRow] = []
    for snr_db in args.snr_db:
        for true_shift in args.true_shifts:
            for name in args.estimators:
                key = (snr_db, true_shift, name)
                detail_rows.append(
                    summarize_errors(
                        snr_db,
                        f"{true_shift:.6g}",
                        name,
                        detail_errors[key],
                        detail_metrics[key],
                        args.outlier_threshold_samples,
                    )
                )

    aggregate_rows: list[SummaryRow] = []
    for snr_db in args.snr_db:
        for name in args.estimators:
            key = (snr_db, name)
            aggregate_rows.append(
                summarize_errors(
                    snr_db,
                    "all",
                    name,
                    aggregate_errors[key],
                    aggregate_metrics[key],
                    args.outlier_threshold_samples,
                )
            )

    return detail_rows, aggregate_rows, example


def row_to_dict(row: SummaryRow) -> dict[str, object]:
    return {
        "snr_db": row.snr_db,
        "true_shift_samples": row.true_shift_samples,
        "estimator": row.estimator,
        "trials": row.trials,
        "bias_samples": row.bias_samples,
        "rmse_samples": row.rmse_samples,
        "std_samples": row.std_samples,
        "p95_abs_error_samples": row.p95_abs_error_samples,
        "outlier_rate": row.outlier_rate,
        "mean_metric": row.mean_metric,
    }


def print_summary(rows: list[SummaryRow]) -> None:
    header = (
        "snr_db  true_shift  estimator   trials  "
        "bias_samp   rmse_samp    std_samp    p95_abs    outlier   metric"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.snr_db:6.1f}  {row.true_shift_samples:>10}  {row.estimator:<10}  "
            f"{row.trials:6d}  {row.bias_samples:10.5f}  {row.rmse_samples:10.5f}  "
            f"{row.std_samples:10.5f}  {row.p95_abs_error_samples:9.5f}  "
            f"{row.outlier_rate:8.4f}  {row.mean_metric:7.4f}"
        )


def write_csv(path: Path, detail_rows: list[SummaryRow], aggregate_rows: list[SummaryRow]) -> None:
    fieldnames = list(row_to_dict(detail_rows[0]).keys()) + ["summary"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in detail_rows:
            out = row_to_dict(row)
            out["summary"] = "per_shift"
            writer.writerow(out)
        for row in aggregate_rows:
            out = row_to_dict(row)
            out["summary"] = "all_shifts"
            writer.writerow(out)


def plot_results(
    detail_rows: list[SummaryRow],
    aggregate_rows: list[SummaryRow],
    example: ExamplePeak,
    oversample: int,
    plot_output: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot") from exc

    estimators = list(dict.fromkeys(row.estimator for row in aggregate_rows))
    snrs = sorted(set(row.snr_db for row in aggregate_rows))

    fig, axes = plt.subplots(3, 1, figsize=(10, 13))
    for estimator in estimators:
        rmse = [
            next(row.rmse_samples for row in aggregate_rows if row.snr_db == snr and row.estimator == estimator)
            for snr in snrs
        ]
        axes[0].plot(snrs, rmse, marker="o", label=estimator)
    axes[0].set_title("eRTM peak estimator RMSE vs SNR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("RMSE (samples)")
    axes[0].grid(True)
    axes[0].legend()

    high_snr = max(snrs)
    shifts = sorted(
        float(row.true_shift_samples)
        for row in detail_rows
        if row.snr_db == high_snr and row.true_shift_samples != "all"
    )
    shifts = sorted(set(shifts))
    for estimator in estimators:
        bias = [
            next(
                row.bias_samples
                for row in detail_rows
                if row.snr_db == high_snr
                and row.true_shift_samples == f"{shift:.6g}"
                and row.estimator == estimator
            )
            for shift in shifts
        ]
        axes[1].plot(shifts, bias, marker="o", label=estimator)
    axes[1].set_title(f"Bias vs true fractional shift at {high_snr:g} dB")
    axes[1].set_xlabel("True shift (samples)")
    axes[1].set_ylabel("Bias (samples)")
    axes[1].grid(True)
    axes[1].legend()

    peak_index = next(iter(example.estimates.values())).peak_index
    signed_peak = signed_bin_from_index(peak_index, example.corr.size)
    offsets = np.arange(-24, 25, dtype=np.int64)
    indices = (peak_index + offsets) % example.corr.size
    x_samples = (float(signed_peak) + offsets.astype(np.float64)) / float(oversample)
    axes[2].plot(x_samples, example.corr[indices], marker=".")
    for estimator, estimate in example.estimates.items():
        axes[2].axvline(estimate.shift_samples, linestyle="--", linewidth=1.1, label=estimator)
    axes[2].axvline(example.true_shift_samples, color="black", linestyle=":", linewidth=1.6, label="true")
    axes[2].set_title(
        f"Example correlation peak: SNR={example.snr_db:g} dB, true={example.true_shift_samples:g} samples"
    )
    axes[2].set_xlabel("Shift (samples)")
    axes[2].set_ylabel("Normalized correlation")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    if plot_output is not None:
        plot_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_output, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    detail_rows, aggregate_rows, example = run_simulation(args)
    print_summary(detail_rows if args.print_detail else aggregate_rows)
    if args.csv is not None:
        write_csv(args.csv, detail_rows, aggregate_rows)
        print(f"\nWrote CSV statistics to {args.csv}")
    if args.plot or args.plot_output is not None:
        plot_results(detail_rows, aggregate_rows, example, args.oversample, args.plot_output)
        if args.plot_output is not None:
            print(f"Wrote plot to {args.plot_output}")


if __name__ == "__main__":
    main()
