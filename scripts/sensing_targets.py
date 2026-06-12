from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OsCfarParams:
    train_doppler: int = 20
    train_range: int = 20
    guard_doppler: int = 10
    guard_range: int = 10
    alpha_db: float = float(10.0 * np.log10(50.0))
    min_range_bin: int = 0
    dc_exclusion_bins: int = 0
    min_power_db: float = 0.0
    max_points: int = 256
    rank_percent: float = 75.0
    suppress_doppler: int = 2
    suppress_range: int = 2


def run_os_cfar_2d(
    rd_db,
    params: OsCfarParams,
    active_row_start=None,
    active_row_stop=None,
    active_col_start=None,
    active_col_stop=None,
    dc_center_row=None,
):
    rd_db = np.asarray(rd_db, dtype=np.float32)
    rows, cols = rd_db.shape
    td = max(0, int(params.train_doppler))
    tr = max(0, int(params.train_range))
    gd = max(0, int(params.guard_doppler))
    gr = max(0, int(params.guard_range))
    alpha_db = float(params.alpha_db)
    alpha = max(1e-12, float(np.power(10.0, alpha_db / 10.0)))
    min_range = max(0, int(params.min_range_bin))
    dc_excl = max(0, int(params.dc_exclusion_bins))
    min_power_db = float(params.min_power_db)
    rank_percent = float(params.rank_percent)
    max_points = max(1, int(params.max_points))
    eps = 1e-12

    outer_h = td + gd
    outer_w = tr + gr
    empty_stats = {
        'noise_min': 0.0,
        'noise_max': 0.0,
        'thresh_min': 0.0,
        'thresh_max': 0.0,
        'power_min_db': min_power_db,
        'invalid_cells': 0,
        'nonfinite_cells': 0,
        'nonpositive_cells': 0,
        'os_rank_index': 0,
        'training_cells': 0,
        'rank_percent': rank_percent,
    }
    if rows == 0 or cols == 0 or rows <= 2 * outer_h or cols <= 2 * outer_w:
        return np.empty((0, 2), dtype=np.int32), 0, 0, "off", empty_stats

    row_start = outer_h if active_row_start is None else max(outer_h, int(active_row_start))
    row_stop = (rows - outer_h) if active_row_stop is None else min(rows - outer_h, int(active_row_stop))
    col_start = outer_w if active_col_start is None else max(outer_w, int(active_col_start))
    col_stop = (cols - outer_w) if active_col_stop is None else min(cols - outer_w, int(active_col_stop))
    if row_start >= row_stop or col_start >= col_stop:
        return np.empty((0, 2), dtype=np.int32), 0, 0, "os-cfar", empty_stats

    power = np.power(np.float64(10.0), rd_db.astype(np.float64) / np.float64(10.0), dtype=np.float64)
    footprint = np.ones((2 * outer_h + 1, 2 * outer_w + 1), dtype=bool)
    footprint[outer_h - gd:outer_h + gd + 1, outer_w - gr:outer_w + gr + 1] = False
    training_cells = int(np.count_nonzero(footprint))
    if training_cells <= 0:
        return np.empty((0, 2), dtype=np.int32), 0, 0, "os-cfar", empty_stats

    rank_index = int(round((np.clip(rank_percent, 0.0, 100.0) / 100.0) * (training_cells - 1)))
    active_rows = row_stop - row_start
    active_cols = col_stop - col_start
    active_rd = rd_db[row_start:row_stop, col_start:col_stop]
    active_power = power[row_start:row_stop, col_start:col_stop]

    candidate_valid = np.ones((active_rows, active_cols), dtype=bool)
    if min_range > 0:
        candidate_valid[:, :min(active_cols, min_range)] = False

    center_row = (rows // 2) if dc_center_row is None else int(dc_center_row)
    center_row_local = center_row - row_start
    if dc_excl > 0 and 0 <= center_row_local < active_rows:
        lo = max(0, center_row_local - dc_excl)
        hi = min(active_rows, center_row_local + dc_excl + 1)
        candidate_valid[lo:hi, :] = False

    if not np.any(candidate_valid):
        stats = dict(empty_stats)
        stats.update({
            'os_rank_index': int(rank_index),
            'training_cells': int(training_cells),
            'evaluated_candidates': 0,
        })
        return np.empty((0, 2), dtype=np.int32), 0, 0, "os-cfar", stats

    candidate_scores = np.where(candidate_valid, active_rd, np.float32(-np.inf)).astype(np.float32, copy=False)
    candidate_total = int(np.count_nonzero(np.isfinite(candidate_scores)))
    candidate_limit = min(candidate_total, max(max_points * 32, 1024))
    if candidate_limit <= 0:
        stats = dict(empty_stats)
        stats.update({
            'os_rank_index': int(rank_index),
            'training_cells': int(training_cells),
            'evaluated_candidates': 0,
        })
        return np.empty((0, 2), dtype=np.int32), 0, 0, "os-cfar", stats

    flat_scores = candidate_scores.reshape(-1)
    partition_k = max(0, flat_scores.size - candidate_limit)
    candidate_flat_idx = np.argpartition(flat_scores, partition_k)[partition_k:]
    candidate_flat_idx = candidate_flat_idx[np.argsort(flat_scores[candidate_flat_idx])[::-1]]

    accepted_points = []
    noise_vals = []
    thresh_vals = []
    invalid_cells = 0
    nonfinite_cells = 0
    nonpositive_cells = 0
    suppressed = np.zeros((active_rows, active_cols), dtype=bool)
    suppression_d = max(0, int(params.suppress_doppler))
    suppression_r = max(0, int(params.suppress_range))

    for flat_idx in candidate_flat_idx.tolist():
        local_row = int(flat_idx) // active_cols
        local_col = int(flat_idx) % active_cols
        if suppressed[local_row, local_col]:
            continue

        cut_db = float(active_rd[local_row, local_col])
        if not np.isfinite(cut_db) or cut_db < min_power_db:
            continue

        row = row_start + local_row
        col = col_start + local_col
        window = power[row - outer_h:row + outer_h + 1, col - outer_w:col + outer_w + 1]
        values = window[footprint]
        values = values[np.isfinite(values)]
        if values.size == 0:
            nonfinite_cells += 1
            invalid_cells += 1
            continue

        local_rank_index = min(rank_index, values.size - 1)
        noise_order = float(np.partition(values, local_rank_index)[local_rank_index])
        if not np.isfinite(noise_order):
            nonfinite_cells += 1
            invalid_cells += 1
            continue
        if noise_order <= eps:
            nonpositive_cells += 1
            invalid_cells += 1
            continue

        threshold = float(alpha * noise_order)
        noise_vals.append(noise_order)
        thresh_vals.append(threshold)
        if float(active_power[local_row, local_col]) <= threshold:
            continue

        accepted_points.append((row, col))
        lo_r = max(0, local_row - suppression_d)
        hi_r = min(active_rows, local_row + suppression_d + 1)
        lo_c = max(0, local_col - suppression_r)
        hi_c = min(active_cols, local_col + suppression_r + 1)
        suppressed[lo_r:hi_r, lo_c:hi_c] = True

    points = np.asarray(accepted_points, dtype=np.int32) if accepted_points else np.empty((0, 2), dtype=np.int32)
    raw_hits = int(points.shape[0])
    if points.shape[0] > max_points:
        values = rd_db[points[:, 0], points[:, 1]]
        order = np.argsort(values)[::-1][:max_points]
        points = points[order]
    shown_hits = int(points.shape[0])

    stats = {
        'noise_min': float(np.min(noise_vals)) if noise_vals else 0.0,
        'noise_max': float(np.max(noise_vals)) if noise_vals else 0.0,
        'thresh_min': float(np.min(thresh_vals)) if thresh_vals else 0.0,
        'thresh_max': float(np.max(thresh_vals)) if thresh_vals else 0.0,
        'power_min_db': min_power_db,
        'invalid_cells': invalid_cells,
        'nonfinite_cells': nonfinite_cells,
        'nonpositive_cells': nonpositive_cells,
        'os_rank_index': int(rank_index),
        'training_cells': int(training_cells),
        'rank_percent': rank_percent,
        'suppress_d': int(params.suppress_doppler),
        'suppress_r': int(params.suppress_range),
        'evaluated_candidates': int(candidate_limit),
    }
    return points, raw_hits, shown_hits, "os-cfar", stats


def cluster_detected_targets(
    target_points,
    rd_data,
    point_strengths_db=None,
    *,
    eps_doppler: int = 2,
    eps_range: int = 2,
    min_samples: int = 1,
):
    if target_points is None:
        return []

    points = np.asarray(target_points, dtype=np.int32)
    if points.size == 0 or rd_data is None:
        return []

    rd_data = np.asarray(rd_data)
    rows, cols = rd_data.shape
    in_bounds = (
        (points[:, 0] >= 0)
        & (points[:, 0] < rows)
        & (points[:, 1] >= 0)
        & (points[:, 1] < cols)
    )
    points = points[in_bounds]
    if point_strengths_db is not None:
        point_strengths_db = np.asarray(point_strengths_db, dtype=np.float32)
        if point_strengths_db.shape[0] != in_bounds.shape[0]:
            point_strengths_db = None
        else:
            point_strengths_db = point_strengths_db[in_bounds]
    if points.size == 0:
        return []

    n_points = points.shape[0]
    eps_d = max(0, int(eps_doppler))
    eps_r = max(0, int(eps_range))
    min_pts = max(1, int(min_samples))

    diff_d = np.abs(points[:, None, 0] - points[None, :, 0]) <= eps_d
    diff_r = np.abs(points[:, None, 1] - points[None, :, 1]) <= eps_r
    neighbor_mask = diff_d & diff_r
    neighbor_lists = [np.flatnonzero(neighbor_mask[idx]) for idx in range(n_points)]

    labels = np.full(n_points, -2, dtype=np.int32)  # -2=unvisited, -1=noise
    cluster_id = 0

    for seed_idx in range(n_points):
        if labels[seed_idx] != -2:
            continue

        seed_neighbors = neighbor_lists[seed_idx]
        if seed_neighbors.size < min_pts:
            labels[seed_idx] = -1
            continue

        labels[seed_idx] = cluster_id
        queue = list(seed_neighbors.tolist())
        queue_pos = 0
        while queue_pos < len(queue):
            nbr_idx = int(queue[queue_pos])
            queue_pos += 1
            if labels[nbr_idx] == -1:
                labels[nbr_idx] = cluster_id
            if labels[nbr_idx] != -2:
                continue
            labels[nbr_idx] = cluster_id
            nbr_neighbors = neighbor_lists[nbr_idx]
            if nbr_neighbors.size >= min_pts:
                queue.extend(nbr_neighbors.tolist())
        cluster_id += 1

    clusters = []
    for cur_cluster_id in range(cluster_id):
        cluster_indices = np.flatnonzero(labels == cur_cluster_id)
        if cluster_indices.size == 0:
            continue
        clusters.append(_build_cluster_summary(points, rd_data, point_strengths_db, cluster_indices))

    clusters.sort(key=lambda item: item['peak_strength_db'], reverse=True)
    return clusters


def build_direct_targets(points, rd_data, point_strengths_db=None):
    if points is None:
        return []

    points = np.asarray(points, dtype=np.int32)
    if points.size == 0 or rd_data is None:
        return []

    rd_data = np.asarray(rd_data)
    rows, cols = rd_data.shape
    in_bounds = (
        (points[:, 0] >= 0)
        & (points[:, 0] < rows)
        & (points[:, 1] >= 0)
        & (points[:, 1] < cols)
    )
    points = points[in_bounds]
    if point_strengths_db is not None:
        point_strengths_db = np.asarray(point_strengths_db, dtype=np.float32)
        if point_strengths_db.shape[0] != in_bounds.shape[0]:
            point_strengths_db = None
        else:
            point_strengths_db = point_strengths_db[in_bounds]
    if points.size == 0:
        return []

    targets = []
    for idx, point in enumerate(points):
        if point_strengths_db is not None:
            strength_db = float(point_strengths_db[idx])
        else:
            strength_db = float(rd_data[int(point[0]), int(point[1])])
        targets.append({
            'peak_doppler_idx': int(point[0]),
            'peak_range_idx': int(point[1]),
            'peak_strength_db': strength_db,
            'cluster_size': 1,
            'centroid_doppler_idx': float(point[0]),
            'centroid_range_idx': float(point[1]),
        })

    targets.sort(key=lambda item: item['peak_strength_db'], reverse=True)
    return targets


def cluster_peak_points(clusters):
    if not clusters:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(
        [
            (int(cluster['peak_doppler_idx']), int(cluster['peak_range_idx']))
            for cluster in clusters
        ],
        dtype=np.int32,
    )


def _build_cluster_summary(points, rd_data, point_strengths_db, cluster_indices):
    cluster_points = points[np.asarray(cluster_indices, dtype=np.int32)]
    if point_strengths_db is not None:
        strengths_db = point_strengths_db[np.asarray(cluster_indices, dtype=np.int32)]
    else:
        strengths_db = rd_data[cluster_points[:, 0], cluster_points[:, 1]]
    peak_idx = int(np.argmax(strengths_db))
    peak_point = cluster_points[peak_idx]
    peak_strength_db = float(strengths_db[peak_idx])
    linear_weights = np.power(10.0, strengths_db.astype(np.float64) / 20.0)
    weight_sum = float(np.sum(linear_weights))
    if weight_sum > 0.0:
        centroid_d = float(np.sum(cluster_points[:, 0] * linear_weights) / weight_sum)
        centroid_r = float(np.sum(cluster_points[:, 1] * linear_weights) / weight_sum)
    else:
        centroid_d = float(np.mean(cluster_points[:, 0]))
        centroid_r = float(np.mean(cluster_points[:, 1]))

    return {
        'peak_doppler_idx': int(peak_point[0]),
        'peak_range_idx': int(peak_point[1]),
        'peak_strength_db': peak_strength_db,
        'cluster_size': int(cluster_points.shape[0]),
        'centroid_doppler_idx': centroid_d,
        'centroid_range_idx': centroid_r,
    }
