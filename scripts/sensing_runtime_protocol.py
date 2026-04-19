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
AGGREGATE_MAGIC_VERSION = 0x41534731  # "ASG1"

FRAME_FORMAT_DENSE_CHANNEL_BUFFER = 0
FRAME_FORMAT_COMPACT_RAW = 1
FRAME_FORMAT_DENSE_RANGE_DOPPLER = 2
FRAME_FORMAT_COMPACT_SPARSE = 3

FLAG_COMPACT_MASK = 1 << 0
FLAG_COMPACT_LOCAL_DELAY_DOPPLER = 1 << 1
FLAG_SKIP_SENSING_FFT = 1 << 2
FLAG_ENABLE_MTI = 1 << 3
FLAG_BISTATIC = 1 << 4
FLAG_AGGREGATED_STREAM = 1 << 5
FLAG_BACKEND_SENSING_PROCESSING = 1 << 6
FLAG_SENSING_METADATA_SIDECAR = 1 << 7
WIRE_DATA_FORMAT_COMPLEX_FLOAT32 = 0
WIRE_DATA_FORMAT_COMPLEX_FLOAT16 = 1

PARAMS_PACKET_STRUCT_V1 = struct.Struct("!4s4s11I")
PARAMS_PACKET_STRUCT_V3 = struct.Struct("!4s4s13I")
PARAMS_PACKET_STRUCT_V4 = struct.Struct("!4s4s14I")
PARAMS_PACKET_STRUCT = struct.Struct("!4s4s17I")
COMPACT_HEADER_STRUCT = struct.Struct("!IIIQ")
AGGREGATE_HEADER_STRUCT = struct.Struct("!IIIIQ")
REQUEST_PACKET_STRUCT = struct.Struct("!4s4si")
SENSING_METADATA_HEADER_STRUCT = struct.Struct("<4s11I9fQ")
AGGREGATE_METADATA_HEADER_STRUCT = struct.Struct("<4sIIIQ")
SENSING_CLUSTER_DTYPE = np.dtype(
    [
        ("peak_doppler_idx", "<i4"),
        ("peak_range_idx", "<i4"),
        ("peak_strength_db", "<f4"),
        ("cluster_size", "<u4"),
        ("centroid_doppler_idx", "<f4"),
        ("centroid_range_idx", "<f4"),
    ]
)


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
    wire_data_format: int = WIRE_DATA_FORMAT_COMPLEX_FLOAT32
    stream_channel_count: int = 1
    stream_channel_mask: int = 1
    backend_os_rank_percent: float = 75.0
    backend_os_suppress_doppler: int = 2
    backend_os_suppress_range: int = 2

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

    def aggregated_stream(self) -> bool:
        return bool(self.flags & FLAG_AGGREGATED_STREAM)

    def backend_processing(self) -> bool:
        return bool(self.flags & FLAG_BACKEND_SENSING_PROCESSING)

    def metadata_sidecar(self) -> bool:
        return bool(self.flags & FLAG_SENSING_METADATA_SIDECAR)

    def wire_format_name(self) -> str:
        if self.wire_data_format == WIRE_DATA_FORMAT_COMPLEX_FLOAT16:
            return "cf16"
        return "cf32"

    def wire_complex_bytes(self) -> int:
        if self.wire_data_format == WIRE_DATA_FORMAT_COMPLEX_FLOAT16:
            return 4
        return 8

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
        if self.aggregated_stream():
            extras.append(f"agg={max(1, int(self.stream_channel_count))}ch")
        if self.backend_processing():
            extras.append("backend")
            extras.append(
                f"os={self.backend_os_rank_percent:.0f}%/"
                f"{self.backend_os_suppress_doppler},"
                f"{self.backend_os_suppress_range}")
        if self.metadata_sidecar():
            extras.append("meta")
        if self.wire_data_format != WIRE_DATA_FORMAT_COMPLEX_FLOAT32:
            extras.append(f"wire={self.wire_format_name()}")
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


@dataclass(frozen=True)
class DecodedSensingMetadata:
    frame_id: int
    cfar_points: np.ndarray
    cfar_hits: int
    cfar_shown_hits: int
    cfar_stats: dict
    target_clusters: list[dict]
    md_spectrum: np.ndarray | None
    md_extent: list[float] | None


def build_params_request(value: int = 0) -> bytes:
    return REQUEST_PACKET_STRUCT.pack(REQ_HEADER, PARAMS_COMMAND, int(value))


def parse_params_packet(data: bytes) -> ViewerRuntimeParams | None:
    if len(data) < PARAMS_PACKET_STRUCT_V1.size:
        return None
    if len(data) >= PARAMS_PACKET_STRUCT.size:
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
            wire_data_format,
            stream_channel_count,
            stream_channel_mask,
            os_cfar_rank_percent_x100,
            os_cfar_suppress_doppler,
            os_cfar_suppress_range,
        ) = PARAMS_PACKET_STRUCT.unpack_from(data)
    elif len(data) >= PARAMS_PACKET_STRUCT_V4.size:
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
            wire_data_format,
            stream_channel_count,
            stream_channel_mask,
        ) = PARAMS_PACKET_STRUCT_V4.unpack_from(data)
        os_cfar_rank_percent_x100 = 7500
        os_cfar_suppress_doppler = 2
        os_cfar_suppress_range = 2
    elif len(data) >= PARAMS_PACKET_STRUCT_V3.size:
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
            stream_channel_count,
            stream_channel_mask,
        ) = PARAMS_PACKET_STRUCT_V3.unpack_from(data)
        wire_data_format = WIRE_DATA_FORMAT_COMPLEX_FLOAT32
        os_cfar_rank_percent_x100 = 7500
        os_cfar_suppress_doppler = 2
        os_cfar_suppress_range = 2
    else:
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
        ) = PARAMS_PACKET_STRUCT_V1.unpack_from(data)
        wire_data_format = WIRE_DATA_FORMAT_COMPLEX_FLOAT32
        stream_channel_count = 1
        stream_channel_mask = 1
        os_cfar_rank_percent_x100 = 7500
        os_cfar_suppress_doppler = 2
        os_cfar_suppress_range = 2
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
        wire_data_format=int(wire_data_format),
        stream_channel_count=max(1, int(stream_channel_count)),
        stream_channel_mask=int(stream_channel_mask) if int(stream_channel_mask) != 0 else 1,
        backend_os_rank_percent=float(int(os_cfar_rank_percent_x100)) / 100.0,
        backend_os_suppress_doppler=max(0, int(os_cfar_suppress_doppler)),
        backend_os_suppress_range=max(0, int(os_cfar_suppress_range)),
    )


def _decode_wire_complex_payload(payload: bytes, expected_count: int, wire_data_format: int) -> np.ndarray:
    if wire_data_format == WIRE_DATA_FORMAT_COMPLEX_FLOAT16:
        scalar = np.frombuffer(payload, dtype=np.float16)
        expected_scalar = expected_count * 2
        if scalar.size != expected_scalar:
            raise ValueError(
                f"Half payload scalar count mismatch: got {scalar.size}, expected {expected_scalar}."
            )
        pairs = scalar.astype(np.float32, copy=False).reshape((expected_count, 2))
        return (pairs[:, 0] + 1j * pairs[:, 1]).astype(np.complex64, copy=False)

    element_size = np.dtype(np.complex64).itemsize
    if len(payload) % element_size != 0:
        raise ValueError(
            f"Dense payload byte size {len(payload)} is not a multiple of complex64 size {element_size}."
        )
    data = np.frombuffer(payload, dtype=np.complex64)
    if data.size != expected_count:
        raise ValueError(
            f"Dense payload complex count mismatch: got {data.size}, expected {expected_count}."
        )
    return data


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
            expected_bytes = re_count * params.wire_complex_bytes()
            if len(data_bytes) != expected_bytes:
                raise ValueError(
                    f"Compact payload byte size mismatch: got {len(data_bytes)}, expected {expected_bytes}."
                )
            if params.compact_mask_hash and mask_hash != params.compact_mask_hash:
                raise ValueError(
                    f"Compact mask hash mismatch: got 0x{mask_hash:08x}, expected 0x{params.compact_mask_hash:08x}."
                )
            matrix = _decode_wire_complex_payload(
                data_bytes,
                expected_re_count,
                params.wire_data_format).reshape((rows, cols))
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
    expected_bytes = expected_count * params.wire_complex_bytes()
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Dense payload byte size mismatch: got {len(payload)}, expected {expected_bytes} ({rows}x{cols})."
        )
    data = _decode_wire_complex_payload(payload, expected_count, params.wire_data_format)
    return DecodedSensingFrame(
        frame_id=int(frame_id_hint),
        matrix=data.reshape((rows, cols)),
    )


def _expand_channel_ids(channel_count: int, channel_mask: int) -> list[int]:
    if channel_mask:
        channel_ids = [bit for bit in range(32) if channel_mask & (1 << bit)]
        if len(channel_ids) == channel_count:
            return channel_ids
    return list(range(channel_count))


def decode_aggregate_sensing_payload(
    frame_id_hint: int,
    payload: bytes,
    params: ViewerRuntimeParams,
) -> tuple[int, list[tuple[int, DecodedSensingFrame]]]:
    if len(payload) < AGGREGATE_HEADER_STRUCT.size:
        raise ValueError("Aggregate sensing payload is shorter than the aggregate header.")

    (
        magic_version,
        channel_count,
        channel_payload_bytes,
        channel_mask,
        frame_start_symbol_index,
    ) = AGGREGATE_HEADER_STRUCT.unpack_from(payload)
    if magic_version != AGGREGATE_MAGIC_VERSION:
        raise ValueError(
            f"Unexpected aggregate magic 0x{magic_version:08x}; expected 0x{AGGREGATE_MAGIC_VERSION:08x}."
        )

    channel_count = int(channel_count)
    channel_payload_bytes = int(channel_payload_bytes)
    if channel_count <= 0 or channel_payload_bytes <= 0:
        raise ValueError("Aggregate sensing header describes an empty channel payload.")

    expected_size = AGGREGATE_HEADER_STRUCT.size + channel_count * channel_payload_bytes
    if len(payload) != expected_size:
        raise ValueError(
            f"Aggregate payload byte size mismatch: got {len(payload)}, expected {expected_size}."
        )

    channel_ids = _expand_channel_ids(channel_count, int(channel_mask))
    decoded_frames: list[tuple[int, DecodedSensingFrame]] = []
    offset = AGGREGATE_HEADER_STRUCT.size
    inner_frame_hint = int(frame_start_symbol_index & 0xFFFFFFFF)
    for ch_id in channel_ids:
        ch_payload = payload[offset:offset + channel_payload_bytes]
        decoded = decode_sensing_payload(inner_frame_hint, ch_payload, params)
        if not decoded.used_compact_header:
            decoded = DecodedSensingFrame(
                frame_id=int(frame_start_symbol_index),
                matrix=decoded.matrix,
                compact_mask_hash=decoded.compact_mask_hash,
                used_compact_header=False,
            )
        decoded_frames.append((int(ch_id), decoded))
        offset += channel_payload_bytes

    return int(frame_start_symbol_index), decoded_frames


def decode_sensing_metadata_payload(payload: bytes) -> DecodedSensingMetadata:
    if len(payload) < SENSING_METADATA_HEADER_STRUCT.size:
        raise ValueError("Metadata payload shorter than metadata header")

    (
        magic,
        total_bytes,
        flags,
        cfar_point_count,
        cluster_count,
        md_rows,
        md_cols,
        cfar_hits,
        cfar_shown_hits,
        invalid_cells,
        nonfinite_cells,
        nonpositive_cells,
        noise_min,
        noise_max,
        thresh_min,
        thresh_max,
        power_min_db,
        md_t0,
        md_t1,
        md_f0,
        md_f1,
        frame_start_symbol_index,
    ) = SENSING_METADATA_HEADER_STRUCT.unpack_from(payload)

    if magic != b"SMD1":
        raise ValueError(f"Unexpected metadata magic {magic!r}")
    total_bytes = int(total_bytes)
    if total_bytes != len(payload):
        raise ValueError(
            f"Metadata payload size mismatch: got {len(payload)}, expected {total_bytes}"
        )

    offset = SENSING_METADATA_HEADER_STRUCT.size
    cfar_points = np.empty((0, 2), dtype=np.int32)
    if cfar_point_count:
        point_bytes = int(cfar_point_count) * 8
        point_slice = payload[offset:offset + point_bytes]
        if len(point_slice) != point_bytes:
            raise ValueError("Metadata CFAR point payload truncated")
        cfar_points = np.frombuffer(point_slice, dtype="<i4").reshape((-1, 2)).astype(np.int32, copy=False)
        offset += point_bytes

    target_clusters: list[dict] = []
    if cluster_count:
        cluster_bytes = int(cluster_count) * SENSING_CLUSTER_DTYPE.itemsize
        cluster_slice = payload[offset:offset + cluster_bytes]
        if len(cluster_slice) != cluster_bytes:
            raise ValueError("Metadata cluster payload truncated")
        cluster_arr = np.frombuffer(cluster_slice, dtype=SENSING_CLUSTER_DTYPE)
        target_clusters = [
            {
                "peak_doppler_idx": int(item["peak_doppler_idx"]),
                "peak_range_idx": int(item["peak_range_idx"]),
                "peak_strength_db": float(item["peak_strength_db"]),
                "cluster_size": int(item["cluster_size"]),
                "centroid_doppler_idx": float(item["centroid_doppler_idx"]),
                "centroid_range_idx": float(item["centroid_range_idx"]),
            }
            for item in cluster_arr
        ]
        offset += cluster_bytes

    md_spectrum = None
    md_extent = None
    total_md_values = int(md_rows) * int(md_cols)
    if total_md_values > 0:
        md_bytes = total_md_values * np.dtype("<f4").itemsize
        md_slice = payload[offset:offset + md_bytes]
        if len(md_slice) != md_bytes:
            raise ValueError("Metadata micro-Doppler payload truncated")
        md_spectrum = np.frombuffer(md_slice, dtype="<f4").reshape((int(md_rows), int(md_cols)))
        md_extent = [float(md_t0), float(md_t1), float(md_f0), float(md_f1)]

    cfar_stats = {
        "noise_min": float(noise_min),
        "noise_max": float(noise_max),
        "thresh_min": float(thresh_min),
        "thresh_max": float(thresh_max),
        "power_min_db": float(power_min_db),
        "invalid_cells": int(invalid_cells),
        "nonfinite_cells": int(nonfinite_cells),
        "nonpositive_cells": int(nonpositive_cells),
        "backend_flags": int(flags),
    }
    return DecodedSensingMetadata(
        frame_id=int(frame_start_symbol_index),
        cfar_points=cfar_points,
        cfar_hits=int(cfar_hits),
        cfar_shown_hits=int(cfar_shown_hits),
        cfar_stats=cfar_stats,
        target_clusters=target_clusters,
        md_spectrum=md_spectrum,
        md_extent=md_extent,
    )


def decode_aggregate_sensing_metadata_payload(
    payload: bytes,
) -> tuple[int, list[tuple[int, DecodedSensingMetadata]]]:
    if len(payload) < AGGREGATE_METADATA_HEADER_STRUCT.size:
        raise ValueError("Aggregate metadata payload shorter than header")

    magic, channel_count, channel_mask, _, frame_start_symbol_index = (
        AGGREGATE_METADATA_HEADER_STRUCT.unpack_from(payload)
    )
    if magic != b"ASM1":
        raise ValueError(f"Unexpected aggregate metadata magic {magic!r}")

    channel_ids = _expand_channel_ids(int(channel_count), int(channel_mask))
    decoded: list[tuple[int, DecodedSensingMetadata]] = []
    offset = AGGREGATE_METADATA_HEADER_STRUCT.size
    for ch_id in channel_ids:
        if offset + SENSING_METADATA_HEADER_STRUCT.size > len(payload):
            raise ValueError("Aggregate metadata payload truncated before per-channel header")
        (_, total_bytes, *_) = SENSING_METADATA_HEADER_STRUCT.unpack_from(payload, offset)
        total_bytes = int(total_bytes)
        channel_payload = payload[offset:offset + total_bytes]
        if len(channel_payload) != total_bytes:
            raise ValueError("Aggregate metadata channel payload truncated")
        meta = decode_sensing_metadata_payload(channel_payload)
        decoded.append((int(ch_id), meta))
        offset += total_bytes

    return int(frame_start_symbol_index), decoded
