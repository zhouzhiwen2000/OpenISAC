"""Bounded slow-time history that preserves elapsed time across missing CPIs."""
from __future__ import annotations

from collections import deque
from itertools import repeat


class TimestampedSensingHistory(deque):
    """Retain observed samples and zero-fill missing slow-time positions.

    Zero fill represents missing observations, not measured zero echoes. It keeps
    the STFT clock intact without interpolating unknown target phase. Timestamp
    rewind and sampling-geometry changes start a new history. Gap work is bounded
    by the retained history length, even after a long interruption.
    """

    def __init__(self, maxlen: int):
        super().__init__(maxlen=maxlen)
        self._next_symbol = None
        self._geometry = None

    def clear(self):
        super().clear()
        self._next_symbol = None
        self._geometry = None

    def append_frame(self, signal, first_symbol: int, params) -> int:
        """Append a timestamped frame; return the number of missing samples."""
        stride = max(1, int(params.sensing_symbol_stride))
        geometry = (params.sample_rate_hz, params.ofdm_fft_size, params.cp_length, stride)
        first_symbol = int(first_symbol)
        if self._geometry != geometry or (self._next_symbol is not None and first_symbol < self._next_symbol):
            self.clear()
        gap = 0
        if self._next_symbol is not None:
            # Small non-grid timestamp jumps are rounded to the nearest slow-time
            # position rather than repeatedly clearing an otherwise useful image.
            gap = max(0, (first_symbol - self._next_symbol + stride // 2) // stride)
            self.extend(repeat(0j, min(gap, self.maxlen)))
        self.extend(signal)
        self._next_symbol = first_symbol + len(signal) * stride
        self._geometry = geometry
        return gap
