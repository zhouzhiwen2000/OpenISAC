from __future__ import annotations

import numpy as np


def normalize_channel_bias_vector(values, expected_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex64).ravel()
    expected = max(1, int(expected_len))
    if arr.size < expected:
        arr = np.pad(arr, (0, expected - arr.size), constant_values=1.0 + 0.0j)
    elif arr.size > expected:
        arr = arr[:expected]
    arr[np.abs(arr) < 1e-12] = 1.0 + 0.0j
    return arr
