from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


_EPS = 1e-12
_PSF_CACHE: dict[tuple, "CleanPsf"] = {}


@dataclass(frozen=True)
class CleanParams:
    loop_gain: float = 1.0
    max_targets: int = 64
    min_power_db: float = 0.0
    min_range_bin: int = 0
    dc_exclusion_bins: int = 0
    psf_threshold_db: float = -35.0
    min_half_width: int = 3


@dataclass(frozen=True)
class CleanPsf:
    kernel: np.ndarray
    support_half_doppler: int
    support_half_range: int
    psf_rows: int
    psf_cols: int


@dataclass(frozen=True)
class CleanStats:
    loop_gain: float
    max_targets: int
    power_min_db: float
    psf_half_doppler: int
    psf_half_range: int
    psf_rows: int
    psf_cols: int
    peak_max_db: float
    peak_min_db: float
    residual_peak_db: float
    stop_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_detection_views(
    rd_data,
    display_range_bins: int,
    display_doppler_bins: int,
    downsample: int,
    pad_doppler_bins: int = 0,
    pad_range_bins: int = 0,
) -> tuple:
    rows, cols = rd_data.shape
    ds = max(1, int(downsample))
    extra_rows = max(0, int(pad_doppler_bins)) * ds
    extra_cols = max(0, int(pad_range_bins)) * ds

    display_col_stop = min(max(1, int(display_range_bins)), cols)
    center_idx = rows // 2
    display_row_start = max(0, center_idx - max(1, int(display_doppler_bins)) // 2)
    display_row_stop = min(rows, display_row_start + max(1, int(display_doppler_bins)))

    display_row_indices = np.arange(display_row_start, display_row_stop, ds, dtype=np.int32)
    display_col_indices = np.arange(0, display_col_stop, ds, dtype=np.int32)
    if display_row_indices.size == 0 or display_col_indices.size == 0:
        empty = rd_data[:0, :0]
        return empty, empty, 0, 0

    wide_row_start = max(0, display_row_start - extra_rows)
    wide_row_stop = min(rows, display_row_stop + extra_rows)
    wide_row_indices = np.arange(wide_row_start, wide_row_stop, ds, dtype=np.int32)
    raw_wide_col_indices = np.arange(-extra_cols, display_col_stop + extra_cols, ds, dtype=np.int32)
    wide_col_indices = np.mod(raw_wide_col_indices, cols).astype(np.int32, copy=False)

    display_view = rd_data[np.ix_(display_row_indices, display_col_indices)]
    wide_view = rd_data[np.ix_(wide_row_indices, wide_col_indices)]
    row_offset = int(np.searchsorted(wide_row_indices, display_row_indices[0]))
    col_offset = int(np.count_nonzero(raw_wide_col_indices < 0))
    return display_view, wide_view, row_offset, col_offset


def estimate_clean_padding(
    raw_rows: int,
    raw_cols: int,
    range_fft_size: int,
    doppler_fft_size: int,
    downsample: int,
    enable_range_window: bool,
    enable_doppler_window: bool,
    params: CleanParams,
) -> tuple[int, int]:
    psf = _get_clean_psf(
        raw_rows,
        raw_cols,
        range_fft_size,
        doppler_fft_size,
        downsample,
        enable_range_window,
        enable_doppler_window,
        params.psf_threshold_db,
        params.min_half_width,
    )
    return psf.support_half_doppler, psf.support_half_range


def run_local_psf_clean(
    rd_complex,
    params: CleanParams,
    *,
    raw_rows: int,
    raw_cols: int,
    range_fft_size: int,
    doppler_fft_size: int,
    downsample: int,
    enable_range_window: bool,
    enable_doppler_window: bool,
    active_row_start: int | None = None,
    active_row_stop: int | None = None,
    active_col_start: int | None = None,
    active_col_stop: int | None = None,
    dc_center_row: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int, str, dict]:
    rd_complex = np.asarray(rd_complex, dtype=np.complex64)
    rows, cols = rd_complex.shape
    psf = _get_clean_psf(
        raw_rows,
        raw_cols,
        range_fft_size,
        doppler_fft_size,
        downsample,
        enable_range_window,
        enable_doppler_window,
        params.psf_threshold_db,
        params.min_half_width,
    )
    empty_stats = CleanStats(
        loop_gain=float(params.loop_gain),
        max_targets=max(1, int(params.max_targets)),
        power_min_db=float(params.min_power_db),
        psf_half_doppler=psf.support_half_doppler,
        psf_half_range=psf.support_half_range,
        psf_rows=psf.psf_rows,
        psf_cols=psf.psf_cols,
        peak_max_db=0.0,
        peak_min_db=0.0,
        residual_peak_db=0.0,
        stop_reason="off",
    )

    if rows == 0 or cols == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            0,
            0,
            "off",
            empty_stats.to_dict(),
        )

    residual = np.array(rd_complex, dtype=np.complex64, copy=True)
    row_start = 0 if active_row_start is None else max(0, int(active_row_start))
    row_stop = rows if active_row_stop is None else min(rows, int(active_row_stop))
    col_start = 0 if active_col_start is None else max(0, int(active_col_start))
    col_stop = cols if active_col_stop is None else min(cols, int(active_col_stop))

    valid_mask = np.zeros((rows, cols), dtype=bool)
    if row_start < row_stop and col_start < col_stop:
        valid_mask[row_start:row_stop, col_start:col_stop] = True

    if params.min_range_bin > 0 and col_start < col_stop:
        min_range_stop = min(col_stop, col_start + int(params.min_range_bin))
        valid_mask[:, col_start:min_range_stop] = False

    center_row = (rows // 2) if dc_center_row is None else int(dc_center_row)
    if params.dc_exclusion_bins > 0:
        lo = max(0, center_row - int(params.dc_exclusion_bins))
        hi = min(rows, center_row + int(params.dc_exclusion_bins) + 1)
        valid_mask[lo:hi, :] = False

    if not np.any(valid_mask):
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            0,
            0,
            "clean",
            CleanStats(
                **{
                    **empty_stats.to_dict(),
                    "stop_reason": "fully_masked",
                }
            ).to_dict(),
        )

    chosen_mask = np.zeros((rows, cols), dtype=bool)
    kernel = psf.kernel
    kr = psf.support_half_doppler
    kc = psf.support_half_range
    loop_gain = float(np.clip(params.loop_gain, 1e-3, 1.0))
    max_targets = max(1, int(params.max_targets))
    amplitude_norm = np.sqrt(max(1.0, float(raw_rows) * float(raw_cols)))
    peak_history_db: list[float] = []
    points: list[tuple[int, int]] = []
    point_peak_db: list[float] = []
    stop_reason = "max_targets"
    neg_inf = np.float32(-np.inf)

    def _search_peak(search_mask) -> tuple[int, float, float]:
        if not np.any(search_mask):
            return -1, float("-inf"), float("-inf")
        masked_magnitude = np.where(search_mask, np.abs(residual), neg_inf)
        peak_flat = int(np.argmax(masked_magnitude))
        peak_value = float(masked_magnitude.reshape(-1)[peak_flat])
        if not np.isfinite(peak_value) or peak_value <= 0.0:
            return peak_flat, peak_value, float("-inf")
        peak_db = float(20.0 * np.log10((peak_value / amplitude_norm) + _EPS))
        return peak_flat, peak_value, peak_db

    while len(points) < max_targets:
        search_mask = valid_mask & (~chosen_mask)
        if not np.any(search_mask):
            stop_reason = "search_exhausted"
            break

        peak_flat, peak_value, peak_db = _search_peak(search_mask)
        if not np.isfinite(peak_value) or peak_value <= 0.0:
            stop_reason = "nonfinite_peak"
            break

        # Enforce Min P on the residual before any extraction only for the very
        # first component, so an empty/noisy frame does not yield a false hit.
        if not points and peak_db < float(params.min_power_db):
            stop_reason = "below_min_power"
            break

        row_idx = peak_flat // cols
        col_idx = peak_flat % cols
        points.append((int(row_idx), int(col_idx)))
        peak_history_db.append(peak_db)
        point_peak_db.append(peak_db)
        chosen_mask[row_idx, col_idx] = True

        peak_complex = np.complex64(residual[row_idx, col_idx])
        row_lo = max(0, row_idx - kr)
        row_hi = min(rows, row_idx + kr + 1)
        col_lo = max(0, col_idx - kc)
        col_hi = min(cols, col_idx + kc + 1)

        k_row_lo = kr - (row_idx - row_lo)
        k_row_hi = k_row_lo + (row_hi - row_lo)
        k_col_lo = kc - (col_idx - col_lo)
        k_col_hi = k_col_lo + (col_hi - col_lo)

        residual[row_lo:row_hi, col_lo:col_hi] -= (
            loop_gain * peak_complex * kernel[k_row_lo:k_row_hi, k_col_lo:k_col_hi]
        )

        next_search_mask = valid_mask & (~chosen_mask)
        if not np.any(next_search_mask):
            stop_reason = "search_exhausted"
            break
        _, next_peak_value, next_peak_db = _search_peak(next_search_mask)
        if not np.isfinite(next_peak_value) or next_peak_value <= 0.0:
            stop_reason = "nonfinite_peak"
            break
        if next_peak_db < float(params.min_power_db):
            stop_reason = "residual_below_min_power"
            break

    if points:
        residual_search_mask = valid_mask & (~chosen_mask)
        _, residual_peak, residual_peak_db = _search_peak(residual_search_mask)
        if not np.isfinite(residual_peak) or residual_peak <= 0.0:
            residual_peak_db = 0.0
        stats = CleanStats(
            loop_gain=loop_gain,
            max_targets=max_targets,
            power_min_db=float(params.min_power_db),
            psf_half_doppler=kr,
            psf_half_range=kc,
            psf_rows=psf.psf_rows,
            psf_cols=psf.psf_cols,
            peak_max_db=float(max(peak_history_db)),
            peak_min_db=float(min(peak_history_db)),
            residual_peak_db=residual_peak_db,
            stop_reason=stop_reason,
        )
        point_array = np.asarray(points, dtype=np.int32)
        point_peak_db_array = np.asarray(point_peak_db, dtype=np.float32)
        raw_hits = int(point_array.shape[0])
        shown_hits = raw_hits
        return point_array, point_peak_db_array, raw_hits, shown_hits, "clean", stats.to_dict()

    return (
        np.empty((0, 2), dtype=np.int32),
        np.empty((0,), dtype=np.float32),
        0,
        0,
        "clean",
        CleanStats(
            **{
                **empty_stats.to_dict(),
                "stop_reason": stop_reason,
            }
        ).to_dict(),
    )


def _get_clean_psf(
    raw_rows: int,
    raw_cols: int,
    range_fft_size: int,
    doppler_fft_size: int,
    downsample: int,
    enable_range_window: bool,
    enable_doppler_window: bool,
    psf_threshold_db: float,
    min_half_width: int,
) -> CleanPsf:
    key = (
        max(1, int(raw_rows)),
        max(1, int(raw_cols)),
        max(1, int(range_fft_size)),
        max(1, int(doppler_fft_size)),
        max(1, int(downsample)),
        bool(enable_range_window),
        bool(enable_doppler_window),
        float(psf_threshold_db),
        max(0, int(min_half_width)),
    )
    cached = _PSF_CACHE.get(key)
    if cached is not None:
        return cached

    doppler_resp = _build_axis_response(
        axis_size=key[0],
        fft_size=key[3],
        downsample=key[4],
        use_window=key[6],
        threshold_db=key[7],
        min_half_width=key[8],
        transform="fft",
    )
    range_resp = _build_axis_response(
        axis_size=key[1],
        fft_size=key[2],
        downsample=key[4],
        use_window=key[5],
        threshold_db=key[7],
        min_half_width=key[8],
        transform="ifft",
    )

    kernel = np.outer(doppler_resp, range_resp).astype(np.complex64, copy=False)
    center_value = kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]
    if np.abs(center_value) > _EPS:
        kernel = kernel / center_value
    cached = CleanPsf(
        kernel=kernel.astype(np.complex64, copy=False),
        support_half_doppler=(kernel.shape[0] - 1) // 2,
        support_half_range=(kernel.shape[1] - 1) // 2,
        psf_rows=int(kernel.shape[0]),
        psf_cols=int(kernel.shape[1]),
    )
    _PSF_CACHE[key] = cached
    return cached


def _build_axis_response(
    axis_size: int,
    fft_size: int,
    downsample: int,
    use_window: bool,
    threshold_db: float,
    min_half_width: int,
    transform: str,
) -> np.ndarray:
    axis_size = max(1, int(axis_size))
    fft_size = max(axis_size, int(fft_size))
    downsample = max(1, int(downsample))
    min_half_width = max(0, int(min_half_width))

    if use_window:
        window = np.hamming(axis_size).astype(np.float32, copy=False)
    else:
        window = np.ones(axis_size, dtype=np.float32)
    centered_window = np.fft.ifftshift(window.astype(np.complex64, copy=False))
    if transform == "ifft":
        response = np.fft.fftshift(np.fft.ifft(centered_window, n=fft_size) * fft_size)
    else:
        response = np.fft.fftshift(np.fft.fft(centered_window, n=fft_size))

    center = fft_size // 2
    center_value = response[center]
    if np.abs(center_value) <= _EPS:
        return np.ones((2 * min_half_width + 1,), dtype=np.complex64)
    response = response / center_value

    positive = response[center::downsample]
    positive_db = 20.0 * np.log10(np.abs(positive) + _EPS)
    above_threshold = np.flatnonzero(positive_db >= float(threshold_db))
    support_half = min_half_width
    if above_threshold.size > 0:
        support_half = max(support_half, int(above_threshold[-1]))

    offsets = np.arange(-support_half, support_half + 1, dtype=np.int32) * downsample
    sample_indices = np.clip(center + offsets, 0, fft_size - 1)
    samples = response[sample_indices]
    samples = samples / (samples[support_half] if np.abs(samples[support_half]) > _EPS else 1.0)
    return samples.astype(np.complex64, copy=False)
