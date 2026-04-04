from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np


CTRL_HEADER = b"CTRL"
CMD_HEADER = b"CMD "
REQ_HEADER = b"REQ "
READY_COMMAND = b"RDY "
PARAMS_COMMAND = b"PARM"

COMPACT_MAGIC_VERSION = 0x43534D31  # "CSM1"

FRAME_FORMAT_DENSE_CHANNEL_BUFFER = 0
FRAME_FORMAT_COMPACT_RAW = 1
FRAME_FORMAT_DENSE_RANGE_DOPPLER = 2
FRAME_FORMAT_COMPACT_SPARSE = 3

FLAG_COMPACT_MASK = 1 << 0
FLAG_COMPACT_LOCAL_DELAY_DOPPLER = 1 << 1
FLAG_SKIP_SENSING_FFT = 1 << 2
FLAG_ENABLE_MTI = 1 << 3
FLAG_BISTATIC = 1 << 4

PARAMS_PACKET_STRUCT = struct.Struct("!4s4s11I")
COMPACT_HEADER_STRUCT = struct.Struct("!IIIQ")
REQUEST_PACKET_STRUCT = struct.Struct("!4s4si")


@dataclass(frozen=True)
class ViewerRuntimeParams:
    version: int = 0
    flags: int = 0
    frame_format: int = FRAME_FORMAT_DENSE_CHANNEL_BUFFER
    wire_rows: int = 100
    wire_cols: int = 1024
    active_rows: int = 100
    active_cols: int = 1024
    frame_symbol_period: int = 100
    range_fft_size: int = 1024
    doppler_fft_size: int = 100
    compact_mask_hash: int = 0

    def is_compact_mask(self) -> bool:
        return bool(self.flags & FLAG_COMPACT_MASK)

    def is_compact_raw(self) -> bool:
        return self.frame_format == FRAME_FORMAT_COMPACT_RAW

    def is_dense_channel_buffer(self) -> bool:
        return self.frame_format == FRAME_FORMAT_DENSE_CHANNEL_BUFFER

    def is_dense_range_doppler(self) -> bool:
        return self.frame_format == FRAME_FORMAT_DENSE_RANGE_DOPPLER

    def is_sparse_compact(self) -> bool:
        return self.frame_format == FRAME_FORMAT_COMPACT_SPARSE

    def raw_fft_locally_supported(self) -> bool:
        return self.frame_format in (
            FRAME_FORMAT_DENSE_CHANNEL_BUFFER,
            FRAME_FORMAT_COMPACT_RAW,
        )

    def compact_local_delay_doppler_supported(self) -> bool:
        return bool(self.flags & FLAG_COMPACT_LOCAL_DELAY_DOPPLER)

    def skip_sensing_fft(self) -> bool:
        return bool(self.flags & FLAG_SKIP_SENSING_FFT)

    def max_strd_value(self) -> int:
        return max(1, int(self.frame_symbol_period))

    def max_range_bin(self) -> int:
        if self.is_dense_range_doppler():
            return max(1, int(self.wire_cols))
        if self.raw_fft_locally_supported():
            return max(1, int(self.range_fft_size))
        return max(1, int(self.wire_cols))

    def describe(self) -> str:
        if self.frame_format == FRAME_FORMAT_DENSE_CHANNEL_BUFFER:
            fmt = "dense_ch"
        elif self.frame_format == FRAME_FORMAT_COMPACT_RAW:
            fmt = "compact_raw"
        elif self.frame_format == FRAME_FORMAT_DENSE_RANGE_DOPPLER:
            fmt = "dense_rd"
        elif self.frame_format == FRAME_FORMAT_COMPACT_SPARSE:
            fmt = "compact_sparse"
        else:
            fmt = f"unknown({self.frame_format})"
        extras = []
        if self.is_compact_mask():
            extras.append("compact")
        if self.compact_local_delay_doppler_supported():
            extras.append("regular")
        if self.skip_sensing_fft():
            extras.append("skip=1")
        return (
            f"{fmt} wire={self.wire_rows}x{self.wire_cols} "
            f"active={self.active_rows}x{self.active_cols} "
            f"range_fft={self.range_fft_size} doppler_fft={self.doppler_fft_size}"
            + (f" [{' '.join(extras)}]" if extras else "")
        )


@dataclass(frozen=True)
class DecodedSensingFrame:
    frame_id: int
    matrix: np.ndarray
    compact_mask_hash: int = 0
    used_compact_header: bool = False


def build_params_request(value: int = 0) -> bytes:
    return REQUEST_PACKET_STRUCT.pack(REQ_HEADER, PARAMS_COMMAND, int(value))


def parse_params_packet(data: bytes) -> ViewerRuntimeParams | None:
    if len(data) < PARAMS_PACKET_STRUCT.size:
        return None
    (
        header,
        command,
        version,
        flags,
        frame_format,
        wire_rows,
        wire_cols,
        active_rows,
        active_cols,
        frame_symbol_period,
        range_fft_size,
        doppler_fft_size,
        compact_mask_hash,
    ) = PARAMS_PACKET_STRUCT.unpack_from(data)
    if header != CTRL_HEADER or command != PARAMS_COMMAND:
        return None
    return ViewerRuntimeParams(
        version=int(version),
        flags=int(flags),
        frame_format=int(frame_format),
        wire_rows=int(wire_rows),
        wire_cols=int(wire_cols),
        active_rows=int(active_rows),
        active_cols=int(active_cols),
        frame_symbol_period=int(frame_symbol_period),
        range_fft_size=int(range_fft_size),
        doppler_fft_size=int(doppler_fft_size),
        compact_mask_hash=int(compact_mask_hash),
    )


def decode_sensing_payload(
    frame_id_hint: int,
    payload: bytes,
    params: ViewerRuntimeParams,
) -> DecodedSensingFrame:
    if len(payload) >= COMPACT_HEADER_STRUCT.size:
        magic_version, mask_hash, re_count, frame_start_symbol_index = COMPACT_HEADER_STRUCT.unpack_from(payload)
        if magic_version == COMPACT_MAGIC_VERSION:
            if not params.is_compact_raw():
                raise ValueError(
                    "Received compact sensing payload, but viewer params do not describe compact raw mode yet."
                )
            rows = int(params.active_rows)
            cols = int(params.active_cols)
            if rows <= 0 or cols <= 0:
                raise ValueError("Compact sensing payload received before valid compact dimensions were known.")
            expected_re_count = rows * cols
            if re_count != expected_re_count:
                raise ValueError(
                    f"Compact payload RE count mismatch: got {re_count}, expected {expected_re_count}."
                )
            data_bytes = payload[COMPACT_HEADER_STRUCT.size:]
            expected_bytes = re_count * np.dtype(np.complex64).itemsize
            if len(data_bytes) != expected_bytes:
                raise ValueError(
                    f"Compact payload byte size mismatch: got {len(data_bytes)}, expected {expected_bytes}."
                )
            if params.compact_mask_hash and mask_hash != params.compact_mask_hash:
                raise ValueError(
                    f"Compact mask hash mismatch: got 0x{mask_hash:08x}, expected 0x{params.compact_mask_hash:08x}."
                )
            matrix = np.frombuffer(data_bytes, dtype=np.complex64).reshape((rows, cols))
            return DecodedSensingFrame(
                frame_id=int(frame_start_symbol_index),
                matrix=matrix,
                compact_mask_hash=int(mask_hash),
                used_compact_header=True,
            )

    rows = int(params.wire_rows)
    cols = int(params.wire_cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("Viewer params do not yet describe a valid dense wire shape.")
    expected_count = rows * cols
    data = np.frombuffer(payload, dtype=np.complex64)
    if data.size != expected_count:
        raise ValueError(
            f"Dense payload complex count mismatch: got {data.size}, expected {expected_count} ({rows}x{cols})."
        )
    return DecodedSensingFrame(
        frame_id=int(frame_id_hint),
        matrix=data.reshape((rows, cols)),
    )
