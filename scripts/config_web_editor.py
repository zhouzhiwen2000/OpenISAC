from __future__ import annotations

import argparse
import atexit
import collections
import copy
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml


HTML_TEMPLATE_PATH = Path(__file__).with_suffix(".html")

PROFILING_OPTIONS: dict[str, tuple[str, ...]] = {
    "modulator": ("all", "modulation", "latency", "data_ingest", "sensing_proc", "sensing_process"),
    "demodulator": ("all", "demodulation", "agc", "align"),
}

PROFILING_DESCRIPTIONS: dict[str, str] = {
    "all": "Enable every profiling and diagnostic module for this tab.",
    "modulation": "Show modulator frame processing breakdown and load statistics.",
    "latency": "Enable modulator end-to-end latency tracking. Requires modulation too.",
    "data_ingest": "Profile UDP ingest, LDPC encode, and enqueue cost in the modulator.",
    "sensing_proc": "Profile per-channel sensing RX-side processing in the modulator path.",
    "sensing_process": "Profile sensing output/aggregation work in the modulator path.",
    "demodulation": "Show demodulator per-frame processing breakdown and load statistics.",
    "agc": "Enable AGC-related runtime diagnostics and gain-adjust logs.",
    "align": "Enable runtime alignment diagnostics, including ALGN logs.",
}

FALLBACK_LAYOUT_FIELDS: dict[str, dict[str, Any]] = {
    "sensing_rx_channels": {
        "title": "Sensing RX Channels",
        "field": {
            "type": "mapping_list",
            "key": "sensing_rx_channels",
            "comment": "",
            "item_fields": [
                {"key": "usrp_channel", "comment": "USRP RX channel index for this sensing path", "kind": "int"},
                {"key": "device_args", "comment": "Per-channel USRP args override (optional)", "kind": "string"},
                {"key": "clock_source", "comment": "Per-channel clock source override (optional)", "kind": "string"},
                {"key": "time_source", "comment": "Per-channel time source override (optional)", "kind": "string"},
                {"key": "wire_format_rx", "comment": "Per-channel wire format override (optional)", "kind": "string"},
                {"key": "rx_gain", "comment": "Per-channel RX gain (dB)", "kind": "int"},
                {"key": "alignment", "comment": "Per-channel timing alignment offset (samples)", "kind": "int"},
                {"key": "rx_antenna", "comment": "RX antenna port (e.g. RX1/RX2/TX/RX)", "kind": "string"},
                {
                    "key": "enable_system_delay_estimation",
                    "comment": "If true, this channel runs delay estimation only and disables sensing pipeline",
                    "kind": "bool",
                },
            ],
        },
    },
    "data_resource_blocks": {
        "title": "Resource Preview",
        "field": {
            "type": "mapping_list",
            "key": "data_resource_blocks",
            "comment": "Optional payload / sensing-pilot RE rectangles",
            "allow_omit": True,
            "planner_kind": "payload",
            "item_fields": [
                {
                    "key": "kind",
                    "comment": "Block role: payload or sensing_pilot",
                    "kind": "string",
                    "options": ["payload", "sensing_pilot"],
                },
                {"key": "symbol_start", "comment": "0-based absolute frame symbol index", "kind": "int"},
                {"key": "symbol_count", "comment": "Number of OFDM symbols in this block", "kind": "int"},
                {"key": "subcarrier_start", "comment": "0-based FFT-bin start index", "kind": "int"},
                {"key": "subcarrier_count", "comment": "Number of contiguous subcarriers in this block", "kind": "int"},
            ],
        },
    },
    "sensing_mask_blocks": {
        "title": "Resource Preview",
        "field": {
            "type": "mapping_list",
            "key": "sensing_mask_blocks",
            "comment": "Optional compact sensing RE rectangles",
            "allow_omit": True,
            "planner_kind": "sensing_mask",
            "item_fields": [
                {"key": "symbol_start", "comment": "0-based absolute frame symbol index", "kind": "int"},
                {"key": "symbol_count", "comment": "Number of OFDM symbols in this block", "kind": "int"},
                {"key": "subcarrier_start", "comment": "0-based FFT-bin start index", "kind": "int"},
                {"key": "subcarrier_count", "comment": "Number of contiguous subcarriers in this block", "kind": "int"},
            ],
        },
    },
    "sensing_output_mode": {
        "title": "Sensing FFT",
        "field": {
            "type": "scalar",
            "key": "sensing_output_mode",
            "comment": "Sensing output mode: dense or compact_mask",
        },
    },
    "sensing_symbol_num": {
        "title": "OFDM Frame",
        "field": {
            "type": "scalar",
            "key": "sensing_symbol_num",
            "comment": "Number of symbols used in sensing processing",
        },
    },
    "range_fft_size": {
        "title": "Sensing FFT",
        "field": {
            "type": "scalar",
            "key": "range_fft_size",
            "comment": "Range FFT size",
        },
    },
    "doppler_fft_size": {
        "title": "Sensing FFT",
        "field": {
            "type": "scalar",
            "key": "doppler_fft_size",
            "comment": "Doppler FFT size",
        },
    },
    "profiling_modules": {
        "title": "Runtime / Debug",
        "field": {
            "type": "scalar",
            "key": "profiling_modules",
            "comment": 'Comma-separated modules to profile, or "all"',
        },
    },
}


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_html_page() -> str:
    return HTML_TEMPLATE_PATH.read_text(encoding="utf-8")


SECTION_RE = re.compile(r"^#\s*=+\s*(.*?)\s*=+\s*$")


@dataclass(frozen=True)
class TabConfig:
    name: str
    label: str
    yaml_path: Path
    cwd: Path
    default_command: str
    presets: tuple[dict[str, str], ...]
    sample_candidates: tuple[Path, ...]


def split_inline_comment(text: str) -> tuple[str, str]:
    if "#" not in text:
        return text.rstrip(), ""
    value, comment = text.split("#", 1)
    return value.rstrip(), comment.strip()


def detect_kind(value: Any, fallback: str = "string") -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "flow_list"
    if isinstance(value, str):
        return "string"
    return fallback


def display_unit_meta(key: str) -> tuple[float, str] | None:
    if key == "sample_rate":
        return (1e6, "MHz")
    if key == "bandwidth":
        return (1e6, "MHz")
    if key == "center_freq":
        return (1e9, "GHz")
    return None


def display_comment_override(key: str, comment: str) -> str:
    overrides = {
        "sample_rate": "Baseband sample rate. Displayed in MHz, stored in Hz.",
        "bandwidth": "Analog bandwidth. Displayed in MHz, stored in Hz.",
        "center_freq": "RF center frequency. Displayed in GHz, stored in Hz.",
    }
    return overrides.get(key, comment)


def format_display_value(key: str, value: Any) -> str:
    meta = display_unit_meta(key)
    if meta is None:
        return "" if value is None else str(value)
    scale, _unit = meta
    if value is None or value == "":
        return ""
    try:
        scaled = float(value) / scale
    except Exception:
        return str(value)
    if abs(scaled - round(scaled)) < 1e-9:
        return str(int(round(scaled)))
    return f"{scaled:.6f}".rstrip("0").rstrip(".")


def parse_display_value(key: str, text: str, kind: str) -> Any:
    meta = display_unit_meta(key)
    if meta is None:
        return coerce_scalar(text, kind)
    scale, _unit = meta
    raw = text.strip()
    if not raw:
        return 0.0 if kind == "float" else 0
    value_hz = float(raw) * scale
    if kind == "int":
        return int(round(value_hz))
    return value_hz


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def profiling_options_for_tab(tab_name: str, value: Any) -> list[str]:
    options = list(PROFILING_OPTIONS.get(tab_name, ()))
    extra_tokens = []
    if isinstance(value, str):
        extra_tokens = [token.strip() for token in value.split(",") if token.strip()]
    for token in extra_tokens:
        if token not in options:
            options.append(token)
    return options


def parse_profiling_modules(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if "all" in tokens:
        return ["all"]
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def encode_profiling_modules(tokens: Any, tab_name: str, fallback: Any) -> str:
    if not isinstance(tokens, list):
        return str(fallback or "")
    cleaned: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        text = str(token).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    if "all" in cleaned:
        return "all"
    option_order = profiling_options_for_tab(tab_name, fallback)
    ordered = [token for token in option_order if token in cleaned and token != "all"]
    extras = [token for token in cleaned if token not in ordered and token != "all"]
    ordered.extend(extras)
    return ",".join(ordered)


def parse_layout(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = [{"title": "General", "fields": []}]
    current = sections[0]
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue
        match = SECTION_RE.match(stripped)
        if match:
            title = match.group(1).strip() or "General"
            if current["fields"]:
                current = {"title": title, "fields": []}
                sections.append(current)
            else:
                current["title"] = title
            i += 1
            continue
        if stripped.startswith("#") or raw.startswith(" "):
            i += 1
            continue
        if ":" not in raw:
            i += 1
            continue

        key, rest = raw.split(":", 1)
        key = key.strip()
        value_part, comment = split_inline_comment(rest)
        value_part = value_part.strip()

        if key == "sensing_rx_channels":
            item_fields: list[dict[str, Any]] = []
            seen_item_keys: set[str] = set()
            j = i + 1
            while j < len(lines):
                sub_raw = lines[j]
                sub_stripped = sub_raw.strip()
                if not sub_stripped:
                    j += 1
                    continue
                if not sub_raw.startswith(" "):
                    break
                if sub_stripped.startswith("#"):
                    j += 1
                    continue
                trimmed = sub_stripped[2:].strip() if sub_stripped.startswith("- ") else sub_stripped
                if ":" in trimmed:
                    item_key, item_rest = trimmed.split(":", 1)
                    item_key = item_key.strip()
                    _, item_comment = split_inline_comment(item_rest)
                    if item_key not in seen_item_keys:
                        item_fields.append({
                            "key": item_key,
                            "comment": item_comment,
                        })
                        seen_item_keys.add(item_key)
                j += 1
            current["fields"].append({
                "type": "mapping_list",
                "key": key,
                "comment": comment,
                "item_fields": item_fields,
            })
            i = j
            continue

        if key == "cpu_cores":
            j = i + 1
            if not value_part:
                while j < len(lines):
                    sub_raw = lines[j]
                    sub_stripped = sub_raw.strip()
                    if not sub_stripped:
                        j += 1
                        continue
                    if not sub_raw.startswith(" "):
                        break
                    j += 1
            current["fields"].append({
                "type": "cpu_cores",
                "key": key,
                "comment": comment,
            })
            i = j if not value_part else i + 1
            continue

        if not value_part:
            item_fields = []
            seen_item_keys: set[str] = set()
            saw_mapping_list = False
            j = i + 1
            while j < len(lines):
                sub_raw = lines[j]
                sub_stripped = sub_raw.strip()
                if not sub_stripped:
                    j += 1
                    continue
                if not sub_raw.startswith(" "):
                    break
                if sub_stripped.startswith("#"):
                    j += 1
                    continue
                trimmed = sub_stripped[2:].strip() if sub_stripped.startswith("- ") else sub_stripped
                if ":" in trimmed:
                    item_key, item_rest = trimmed.split(":", 1)
                    item_key = item_key.strip()
                    _, item_comment = split_inline_comment(item_rest)
                    if item_key not in seen_item_keys:
                        item_fields.append({
                            "key": item_key,
                            "comment": item_comment,
                        })
                        seen_item_keys.add(item_key)
                    saw_mapping_list = True
                j += 1
            if saw_mapping_list:
                current["fields"].append({
                    "type": "mapping_list",
                    "key": key,
                    "comment": comment,
                    "item_fields": item_fields,
                    "allow_omit": key == "data_resource_blocks",
                })
                i = j
                continue

        current["fields"].append({
            "type": "flow_list" if value_part.startswith("[") else "scalar",
            "key": key,
            "comment": comment,
        })
        i += 1

    return [section for section in sections if section["fields"]]


def coerce_scalar(text: str, kind: str) -> Any:
    text = text.strip()
    if kind == "bool":
        return text.lower() == "true"
    if kind == "int":
        return int(text) if text else 0
    if kind == "float":
        return float(text) if text else 0.0
    return text


def coerce_flow_list(text: str, item_kind: str) -> list[Any]:
    raw = text.strip()
    if not raw:
        return []
    items = [item.strip() for item in raw.split(",")]
    return [coerce_scalar(item, item_kind) for item in items if item]


def quote_string(value: str) -> str:
    if value == "" or any(ch in value for ch in [":", "#", ",", "[", "]", "{", "}", " "]):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if value is None:
        return '""'
    return quote_string(str(value))


def format_flow_list(values: list[Any]) -> str:
    return "[" + ", ".join(format_scalar(v) for v in values) + "]"


def cpu_binding_rows(tab_name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if tab_name == "modulator":
        rows.extend([
            {"comment": "idx0 (hint 0): _tx_proc", "label": "_tx_proc"},
            {"comment": "idx1 (hint 1): _modulation_proc", "label": "_modulation_proc"},
            {"comment": "idx2 (hint 2): _data_ingest_proc", "label": "_data_ingest_proc"},
        ])
        count = max(0, int_or_zero(data.get("sensing_rx_channel_count", 0)))
        for ch in range(count):
            hint = len(rows)
            rows.append({
                "comment": f"idx{hint} (hint {hint}): SensingChannel::_rx_loop (CH{ch})",
                "label": f"SensingChannel::_rx_loop (CH{ch})",
            })
            hint = len(rows)
            rows.append({
                "comment": f"idx{hint} (hint {hint}): SensingChannel::_sensing_loop (CH{ch})",
                "label": f"SensingChannel::_sensing_loop (CH{ch})",
            })
    else:
        rows.extend([
            {"comment": "idx0 (hint 0): rx_proc", "label": "rx_proc"},
            {"comment": "idx1 (hint 1): process_proc", "label": "process_proc"},
            {"comment": "idx2 (hint 2): sensing_process_proc", "label": "sensing_process_proc"},
            {"comment": "idx3 (hint 3): bit_processing_proc", "label": "bit_processing_proc"},
        ])
    rows.append({"comment": "idx_last: main thread affinity", "label": "main thread affinity"})

    cpu_values = data.get("cpu_cores", []) or []
    values = [int_or_zero(v) for v in cpu_values]
    while len(rows) < len(values):
        slot = len(rows)
        rows.append({"comment": f"extra[{slot}]", "label": f"extra[{slot}]"})
    for index, row in enumerate(rows):
        row["cpu"] = values[index] if index < len(values) else None
    return rows


def unique_cpu_spec(values: list[int]) -> str:
    unique = sorted({int(v) for v in values})
    return ",".join(str(v) for v in unique)


def append_missing_layout_fields(
    layout: list[dict[str, Any]],
    sample_candidates: tuple[Path, ...],
    required_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    existing = {field["key"] for section in layout for field in section["fields"]}
    missing = [key for key in required_keys if key not in existing]
    if not missing:
        return layout

    sample_layout: list[dict[str, Any]] = []
    for candidate in sample_candidates:
        if candidate.exists():
            sample_layout = parse_layout(candidate.read_text(encoding="utf-8"))
            break

    layout_copy = copy.deepcopy(layout)
    for key in missing:
        inserted = False
        for sample_section in sample_layout:
            for sample_field in sample_section["fields"]:
                if sample_field["key"] != key:
                    continue
                for target_section in layout_copy:
                    if target_section["title"] == sample_section["title"]:
                        target_section["fields"].append(copy.deepcopy(sample_field))
                        inserted = True
                        break
                if not inserted:
                    layout_copy.append({
                        "title": sample_section["title"],
                        "fields": [copy.deepcopy(sample_field)],
                    })
                    inserted = True
                break
            if inserted:
                break
        if inserted:
            continue
        fallback = FALLBACK_LAYOUT_FIELDS.get(key)
        if fallback is None:
            continue
        for target_section in layout_copy:
            if target_section["title"] == fallback["title"]:
                target_section["fields"].append(copy.deepcopy(fallback["field"]))
                inserted = True
                break
        if not inserted:
            layout_copy.append({
                "title": fallback["title"],
                "fields": [copy.deepcopy(fallback["field"])],
            })
    return layout_copy


def enrich_mapping_list_layouts(layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layout_copy = copy.deepcopy(layout)
    fallback_item_fields_by_key = {
        field_key: {
            item_field["key"]: item_field
            for item_field in fallback["field"].get("item_fields", [])
        }
        for field_key, fallback in FALLBACK_LAYOUT_FIELDS.items()
        if fallback["field"].get("type") == "mapping_list"
    }
    for section in layout_copy:
        for field in section["fields"]:
            if field.get("type") != "mapping_list":
                continue
            field_key = field.get("key")
            fallback_item_fields = fallback_item_fields_by_key.get(field_key, {})
            merged_item_fields: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            for item_field in field.get("item_fields", []):
                item_key = item_field.get("key")
                if not item_key or item_key in seen_keys:
                    continue
                merged_field = copy.deepcopy(fallback_item_fields.get(item_key, {}))
                merged_field.update(item_field)
                merged_item_fields.append(merged_field)
                seen_keys.add(item_key)
            for item_key, fallback_field in fallback_item_fields.items():
                if item_key in seen_keys:
                    continue
                merged_item_fields.append(copy.deepcopy(fallback_field))
            if field_key in {"data_resource_blocks", "sensing_mask_blocks"}:
                field["allow_omit"] = True
            field["item_fields"] = merged_item_fields
    return layout_copy


def default_sensing_channel_item(
    data: dict[str, Any],
    index: int,
    base_item: dict[str, Any] | None = None,
    preserve_usrp_channel: bool = False,
) -> dict[str, Any]:
    base = base_item if isinstance(base_item, dict) else {}
    fallback_item = {
        "usrp_channel": 1,
        "device_args": "",
        "clock_source": "",
        "time_source": "",
        "wire_format_rx": "",
        "rx_gain": 30,
        "alignment": 63,
        "rx_antenna": "RX2",
        "enable_system_delay_estimation": False,
    }
    base_usrp_channel = int_or_default(base.get("usrp_channel"), fallback_item["usrp_channel"])
    usrp_channel = base_usrp_channel if preserve_usrp_channel else base_usrp_channel + index

    return {
        "usrp_channel": usrp_channel,
        "device_args": str(base.get("device_args", fallback_item["device_args"])),
        "clock_source": str(base.get("clock_source", fallback_item["clock_source"])),
        "time_source": str(base.get("time_source", fallback_item["time_source"])),
        "wire_format_rx": str(base.get("wire_format_rx", fallback_item["wire_format_rx"])),
        "rx_gain": int_or_default(base.get("rx_gain"), fallback_item["rx_gain"]),
        "alignment": int_or_default(base.get("alignment"), fallback_item["alignment"]),
        "rx_antenna": str(base.get("rx_antenna", fallback_item["rx_antenna"])),
        "enable_system_delay_estimation": bool(base.get(
            "enable_system_delay_estimation",
            fallback_item["enable_system_delay_estimation"],
        )),
    }


def default_data_resource_block_item(data: dict[str, Any]) -> dict[str, Any]:
    fft_size = max(1, int_or_default(data.get("fft_size"), 1024))
    num_symbols = max(1, int_or_default(data.get("num_symbols"), 100))
    sync_pos = int_or_default(data.get("sync_pos"), 0)
    data_symbol_candidates = [idx for idx in range(num_symbols) if idx != sync_pos]
    symbol_start = data_symbol_candidates[0] if data_symbol_candidates else 0
    symbol_count = min(4, len(data_symbol_candidates)) if data_symbol_candidates else 1
    subcarrier_count = min(128, fft_size)
    return {
        "kind": "payload",
        "symbol_start": symbol_start,
        "symbol_count": max(1, symbol_count),
        "subcarrier_start": 0,
        "subcarrier_count": max(1, subcarrier_count),
    }


def normalized_data_resource_block_items(items: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append({
            "kind": str(item.get("kind", "payload") or "payload"),
            "symbol_start": int_or_zero(item.get("symbol_start")),
            "symbol_count": int_or_zero(item.get("symbol_count")),
            "subcarrier_start": int_or_zero(item.get("subcarrier_start")),
            "subcarrier_count": int_or_zero(item.get("subcarrier_count")),
        })
    return normalized


def normalized_sensing_channel_items(data: dict[str, Any], items: list[Any]) -> list[dict[str, Any]]:
    count = max(0, int_or_zero(data.get("sensing_rx_channel_count", 0)))
    dict_items = [item for item in items if isinstance(item, dict)]
    if count == 0:
        return []
    base_item = dict_items[0] if dict_items else None
    normalized: list[dict[str, Any]] = []
    for index in range(count):
        if index < len(dict_items):
            normalized.append(default_sensing_channel_item(
                data,
                index,
                dict_items[index],
                preserve_usrp_channel=True,
            ))
            continue
        normalized.append(default_sensing_channel_item(data, index, base_item))
    return normalized


def load_yaml_with_layout(path: Path, fallback_paths: tuple[Path, ...]) -> tuple[dict[str, Any], list[dict[str, Any]], bool, Path]:
    exists = path.exists()
    source = path if exists else next((candidate for candidate in fallback_paths if candidate.exists()), path)
    text = source.read_text(encoding="utf-8") if source.exists() else ""
    data = yaml.safe_load(text) if text.strip() else {}
    if not isinstance(data, dict):
        data = {}
    layout = parse_layout(text)
    for section in layout:
        field_keys = {field.get("key") for field in section.get("fields", [])}
        if "data_resource_blocks" in field_keys or "sensing_mask_blocks" in field_keys:
            section["title"] = "Resource Preview"
    if "sensing_rx_channels" in data and not isinstance(data["sensing_rx_channels"], list):
        data["sensing_rx_channels"] = []
    if "data_resource_blocks" in data and not isinstance(data["data_resource_blocks"], list):
        data["data_resource_blocks"] = []
    else:
        data["data_resource_blocks"] = normalized_data_resource_block_items(data.get("data_resource_blocks", []))
    if "sensing_mask_blocks" in data and not isinstance(data["sensing_mask_blocks"], list):
        data["sensing_mask_blocks"] = []
    if "cpu_cores" in data and not isinstance(data["cpu_cores"], list):
        data["cpu_cores"] = []
    layout = append_missing_layout_fields(
        layout,
        fallback_paths,
        (
            "sensing_rx_channels",
            "data_resource_blocks",
            "sensing_mask_blocks",
            "sensing_output_mode",
            "range_fft_size",
            "doppler_fft_size",
            "sensing_symbol_num",
            "profiling_modules",
        ),
    )
    layout = enrich_mapping_list_layouts(layout)
    known_keys = {field["key"] for section in layout for field in section["fields"]}
    extra_keys = [key for key in data.keys() if key not in known_keys]
    if extra_keys:
        layout.append({
            "title": "Other",
            "fields": [{"type": "scalar", "key": key, "comment": ""} for key in extra_keys],
        })
    return data, layout, exists, source


def build_form_payload(tab_name: str, data: dict[str, Any], layout: list[dict[str, Any]]) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    for section in layout:
        section_payload = {"title": section["title"], "fields": []}
        for field in section["fields"]:
            key = field["key"]
            value = data.get(key)
            if field["type"] == "cpu_cores":
                section_payload["fields"].append({
                    "type": "cpu_cores",
                    "key": key,
                    "comment": field.get("comment", ""),
                })
                continue
            if field["type"] == "mapping_list":
                items = copy.deepcopy(data.get(key, []) or [])
                if not isinstance(items, list):
                    items = []
                if key == "sensing_rx_channels":
                    items = normalized_sensing_channel_items(data, items)
                item_fields = []
                for item_field in field["item_fields"]:
                    sample_value = ""
                    for item in items:
                        if isinstance(item, dict) and item_field["key"] in item:
                            sample_value = item[item_field["key"]]
                            break
                    item_fields.append({
                        "key": item_field["key"],
                        "comment": item_field.get("comment", ""),
                        "display_comment": display_comment_override(item_field["key"], item_field.get("comment", "")),
                        "kind": item_field.get("kind") or detect_kind(sample_value),
                    })
                field_payload = {
                    "type": "mapping_list",
                    "key": key,
                    "comment": field.get("comment", ""),
                    "display_comment": display_comment_override(key, field.get("comment", "")),
                    "item_fields": item_fields,
                    "items": items,
                    "allow_omit": bool(field.get("allow_omit", False)),
                    "planner_kind": field.get("planner_kind", ""),
                }
                if key == "sensing_rx_channels":
                    base_item = items[0] if items else None
                    field_payload["default_item"] = default_sensing_channel_item(
                        data,
                        0,
                        base_item,
                        preserve_usrp_channel=True,
                    )
                if key == "data_resource_blocks":
                    field_payload["default_item"] = default_data_resource_block_item(data)
                    if key not in data:
                        field_payload["mode"] = "legacy"
                    elif items:
                        field_payload["mode"] = "custom"
                    else:
                        field_payload["mode"] = "disabled"
                if key == "sensing_mask_blocks":
                    field_payload["default_item"] = default_data_resource_block_item(data)
                    sensing_mode = str(data.get("sensing_output_mode", "dense") or "dense").strip().lower()
                    field_payload["mode"] = "custom" if sensing_mode == "compact_mask" else "strd"
                section_payload["fields"].append(field_payload)
                continue

            if key == "profiling_modules":
                section_payload["fields"].append({
                    "type": "profiling_modules",
                    "key": key,
                    "comment": field.get("comment", ""),
                    "display_comment": display_comment_override(key, field.get("comment", "")),
                    "options": [
                        {
                            "key": option,
                            "description": PROFILING_DESCRIPTIONS.get(option, ""),
                        }
                        for option in profiling_options_for_tab(tab_name, value)
                    ],
                    "selected": parse_profiling_modules(value),
                })
                continue
            if field["type"] == "flow_list":
                item_kind = detect_kind(value[0], "int") if isinstance(value, list) and value else "int"
                section_payload["fields"].append({
                    "type": "flow_list",
                    "key": key,
                    "comment": field.get("comment", ""),
                    "display_comment": display_comment_override(key, field.get("comment", "")),
                    "kind": item_kind,
                    "value_text": ", ".join(str(v) for v in (value or [])),
                })
                continue

            kind = detect_kind(value)
            unit_meta = display_unit_meta(key)
            section_payload["fields"].append({
                "type": "scalar",
                "key": key,
                "comment": field.get("comment", ""),
                "display_comment": display_comment_override(key, field.get("comment", "")),
                "kind": kind,
                "value": value,
                "value_text": format_display_value(key, value),
                "display_unit": unit_meta[1] if unit_meta is not None else "",
            })
        sections.append(section_payload)

    cpu_values = [int_or_zero(v) for v in data.get("cpu_cores", []) or []]
    return {
        "sections": sections,
        "cpu_cores": {
            "values": cpu_values,
            "rows": cpu_binding_rows(tab_name, data),
        },
    }


def render_yaml(tab_name: str, layout: list[dict[str, Any]], data: dict[str, Any]) -> str:
    lines: list[str] = []
    cpu_rows = cpu_binding_rows(tab_name, data)
    cpu_values = [row.get("cpu") for row in cpu_rows if row.get("cpu") is not None]
    data["cpu_cores"] = [int(v) for v in cpu_values]

    for section in layout:
        lines.append(f"# ===== {section['title']} =====")
        for field in section["fields"]:
            key = field["key"]
            comment = field.get("comment", "")
            suffix = f"  # {comment}" if comment else ""
            if field["type"] == "cpu_cores":
                values = data.get("cpu_cores", []) or []
                if not values:
                    lines.append(f"{key}: []{suffix}")
                else:
                    lines.append(f"{key}:{suffix}")
                    for row in cpu_binding_rows(tab_name, data):
                        if row.get("cpu") is None:
                            continue
                        lines.append(f"  - {row['cpu']}  # {row['comment']}")
                continue

            if field["type"] == "mapping_list":
                if field.get("allow_omit") and key not in data:
                    continue
                items = data.get(key, []) or []
                if not items:
                    lines.append(f"{key}: []{suffix}")
                    continue
                lines.append(f"{key}:{suffix}")
                item_fields = field["item_fields"]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    for index, item_field in enumerate(item_fields):
                        item_key = item_field["key"]
                        item_comment = item_field.get("comment", "")
                        item_suffix = f"  # {item_comment}" if item_comment else ""
                        prefix = "  - " if index == 0 else "    "
                        value = item.get(item_key, "")
                        lines.append(f"{prefix}{item_key}: {format_scalar(value)}{item_suffix}")
                continue

            value = data.get(key)
            if field["type"] == "flow_list":
                lines.append(f"{key}: {format_flow_list(value or [])}{suffix}")
            else:
                lines.append(f"{key}: {format_scalar(value)}{suffix}")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def normalize_payload_data(tab_name: str, layout: list[dict[str, Any]], payload: dict[str, Any], current_data: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(current_data)
    scalars = payload.get("scalars", {})
    mapping_lists = payload.get("mapping_lists", {})
    cpu_values = payload.get("cpu_cores", [])
    if not isinstance(scalars, dict):
        raise RuntimeError("Invalid scalar payload.")
    if not isinstance(mapping_lists, dict):
        raise RuntimeError("Invalid mapping list payload.")

    kind_map: dict[str, tuple[str, str]] = {}
    mapping_layouts: dict[str, list[dict[str, Any]]] = {}
    for section in layout:
        for field in section["fields"]:
            if field["type"] == "mapping_list":
                mapping_layouts[field["key"]] = field["item_fields"]
            elif field["key"] == "profiling_modules":
                kind_map[field["key"]] = ("profiling_modules", "")
            elif field["type"] == "flow_list":
                existing = current_data.get(field["key"], [])
                item_kind = detect_kind(existing[0], "int") if isinstance(existing, list) and existing else "int"
                kind_map[field["key"]] = ("flow_list", item_kind)
            elif field["type"] == "scalar":
                kind_map[field["key"]] = (field.get("kind") or detect_kind(current_data.get(field["key"])), "")

    for key, raw in scalars.items():
        field_kind, extra = kind_map.get(key, ("string", ""))
        if field_kind == "profiling_modules":
            result[key] = encode_profiling_modules(raw, tab_name, current_data.get(key, ""))
        elif field_kind == "flow_list":
            result[key] = coerce_flow_list(str(raw), extra)
        elif field_kind == "bool":
            result[key] = bool(raw)
        else:
            result[key] = parse_display_value(key, str(raw), field_kind)

    for key, raw_mapping in mapping_lists.items():
        item_layout = mapping_layouts.get(key, [])
        mode = "custom"
        items = raw_mapping
        if isinstance(raw_mapping, dict):
            mode_raw = raw_mapping.get("mode", "custom")
            mode = str(mode_raw)
            items = raw_mapping.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError(f"Invalid mapping list for {key}.")
        normalized_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized_item: dict[str, Any] = {}
            for item_field in item_layout:
                item_key = item_field["key"]
                kind = item_field.get("kind") or detect_kind(item.get(item_key))
                raw = item.get(item_key, "")
                if kind == "bool":
                    normalized_item[item_key] = bool(raw)
                else:
                    normalized_item[item_key] = coerce_scalar(str(raw), kind)
            normalized_items.append(normalized_item)
        if key == "data_resource_blocks":
            if mode == "legacy":
                result.pop(key, None)
            elif mode == "disabled":
                result[key] = []
            else:
                result[key] = normalized_data_resource_block_items(normalized_items)
            continue
        if key == "sensing_mask_blocks":
            if mode == "custom":
                result[key] = normalized_items
                result["sensing_output_mode"] = "compact_mask"
            else:
                result.pop(key, None)
                result["sensing_output_mode"] = "dense"
            continue
        result[key] = normalized_items

    if not isinstance(cpu_values, list):
        raise RuntimeError("Invalid cpu_cores payload.")
    result["cpu_cores"] = [int_or_zero(v) for v in cpu_values if str(v).strip() != ""]
    if tab_name == "modulator":
        channels = result.get("sensing_rx_channels", []) or []
        if not isinstance(channels, list):
            channels = []
        result["sensing_rx_channels"] = normalized_sensing_channel_items(result, channels)
    return result


def privileged_command(command: list[str], sudo_password: str = "") -> tuple[list[str], str | None]:
    if os.geteuid() == 0:
        return command, None
    if sudo_password:
        return ["sudo", "-S", "-p", ""] + command, sudo_password + "\n"
    return ["sudo", "-n"] + command, None


class ProcessController:
    def __init__(self, tab_name: str, cwd: Path, default_command: str, isolate_script: Path) -> None:
        self._tab_name = tab_name
        self._cwd = cwd
        self._default_command = default_command
        self._current_command = default_command
        self._process: subprocess.Popen[str] | None = None
        self._returncode: int | None = None
        self._logs: collections.deque[str] = collections.deque(maxlen=4000)
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cpu_spec = ""
        self._unit_name = f"openisac-{tab_name}"
        self._isolate_script = isolate_script
        self._isolate_enabled = True

    def _cleanup_unit(self, sudo_password: str = "") -> None:
        self._run_checked(
            ["systemctl", "stop", self._unit_name],
            ignore_failure=True,
            sudo_password=sudo_password,
        )
        self._run_checked(
            ["systemctl", "reset-failed", self._unit_name],
            ignore_failure=True,
            sudo_password=sudo_password,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            if self._process is not None and not running:
                self._returncode = self._process.poll()
                self._process = None
            pid = self._process.pid if self._process is not None else None
            return {
                "ok": True,
                "running": running,
                "pid": pid,
                "returncode": self._returncode,
                "cwd": str(self._cwd),
                "command": self._current_command,
                "cpu_spec": self._cpu_spec,
                "isolate_enabled": self._isolate_enabled,
                "unit_name": self._unit_name,
                "logs": list(self._logs),
            }

    def start(
        self,
        command: str,
        cpu_values: list[int],
        isolate_cpu: bool,
        override_cpu_spec: str | None = None,
        sudo_password: str = "",
    ) -> dict[str, Any]:
        command = command.strip() or self._default_command
        default_cpu_spec = unique_cpu_spec(cpu_values)
        cpu_spec = (override_cpu_spec or "").strip() if override_cpu_spec else default_cpu_spec
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"{self._tab_name} is already running.")
            self._logs.clear()
            self._logs.append(f"$ {command}")
            self._returncode = None
            self._current_command = command
            self._cpu_spec = cpu_spec if isolate_cpu else ""
            self._isolate_enabled = isolate_cpu

        if isolate_cpu and cpu_spec:
            self._run_checked(
                [str(self._isolate_script), cpu_spec],
                "CPU isolation setup failed",
                sudo_password=sudo_password,
            )
        self._cleanup_unit(sudo_password=sudo_password)
        run_cmd = [
            "systemd-run",
            "--unit",
            self._unit_name,
            "--collect",
            "--slice=rt-workload.slice",
            "--pipe",
            "--quiet",
            "-p",
            f"WorkingDirectory={self._cwd}",
        ]
        if isolate_cpu and cpu_spec:
            run_cmd += ["-p", f"AllowedCPUs={cpu_spec}"]
        run_cmd += ["/bin/bash", "-lc", command]

        with self._lock:
            self._process = self._popen_checked(run_cmd, sudo_password)
            self._reader_thread = threading.Thread(target=self._read_output, args=(self._process,), daemon=True)
            self._reader_thread.start()
        return self.snapshot()

    def stop(self, sudo_password: str = "") -> dict[str, Any]:
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
        self._cleanup_unit(sudo_password=sudo_password)
        if process is not None:
            for _ in range(30):
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    raise RuntimeError("Failed to terminate privileged runtime client.") from exc
        return self.snapshot()

    def reset_isolation(self, sudo_password: str = "") -> dict[str, Any]:
        self._run_checked(
            [str(self._isolate_script), "reset"],
            "CPU isolation reset failed",
            sudo_password=sudo_password,
        )
        return self.snapshot()

    def stop_if_running(self) -> None:
        try:
            self.stop()
        except Exception:
            return

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self._lock:
                self._logs.append(line.rstrip("\n"))
        process.wait()
        with self._lock:
            self._returncode = process.returncode
            if self._process is process:
                self._process = None
            self._logs.append(f"[manager] exited with code {process.returncode}")

    def _popen_checked(self, command: list[str], sudo_password: str = "") -> subprocess.Popen[str]:
        full_command, sudo_input = privileged_command(command, sudo_password)
        process = subprocess.Popen(
            full_command,
            cwd=self._cwd,
            stdin=subprocess.PIPE if sudo_input is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        if sudo_input is not None and process.stdin is not None:
            try:
                process.stdin.write(sudo_input)
                process.stdin.flush()
            finally:
                process.stdin.close()
        return process

    def _run_checked(
        self,
        command: list[str],
        message: str = "",
        ignore_failure: bool = False,
        sudo_password: str = "",
    ) -> None:
        full_command, sudo_input = privileged_command(command, sudo_password)
        result = subprocess.run(
            full_command,
            cwd=self._cwd,
            input=sudo_input,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 or ignore_failure:
            if result.stdout.strip():
                with self._lock:
                    self._logs.append(result.stdout.strip())
            return
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"{message or 'Command failed'}: {detail}")


class ConfigEditorApp:
    def __init__(self, repo_root: Path, build_dir: Path) -> None:
        isolate_script = repo_root / "scripts" / "isolate_cpus.bash"
        self.repo_root = repo_root
        self.build_dir = build_dir
        self.tabs: dict[str, TabConfig] = {
            "modulator": TabConfig(
                name="modulator",
                label="Modulator",
                yaml_path=build_dir / "Modulator.yaml",
                cwd=build_dir,
                default_command="./OFDMModulator",
                presets=(
                    {"label": "CPU Modulator", "command": "./OFDMModulator"},
                ),
                sample_candidates=(
                    repo_root / "config" / "Modulator_X310.yaml",
                    repo_root / "config" / "Modulator_B210.yaml",
                ),
            ),
            "demodulator": TabConfig(
                name="demodulator",
                label="Demodulator",
                yaml_path=build_dir / "Demodulator.yaml",
                cwd=build_dir,
                default_command="./OFDMDemodulator",
                presets=(
                    {"label": "CPU Demodulator", "command": "./OFDMDemodulator"},
                ),
                sample_candidates=(
                    repo_root / "config" / "Demodulator_X310.yaml",
                    repo_root / "config" / "Demodulator_B210.yaml",
                ),
            ),
        }
        self.processes = {
            name: ProcessController(name, tab.cwd, tab.default_command, isolate_script)
            for name, tab in self.tabs.items()
        }

    def tab_config(self, name: str) -> TabConfig:
        try:
            return self.tabs[name]
        except KeyError as exc:
            raise RuntimeError(f"Unknown tab '{name}'.") from exc

    def load_config(self, name: str) -> dict[str, Any]:
        tab = self.tab_config(name)
        data, layout, exists, source = load_yaml_with_layout(tab.yaml_path, tab.sample_candidates)
        mtime = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tab.yaml_path.stat().st_mtime))
            if tab.yaml_path.exists()
            else None
        )
        form_payload = build_form_payload(name, data, layout)
        return {
            "ok": True,
            "path": str(tab.yaml_path),
            "exists": exists,
            "source_path": str(source),
            "mtime": mtime,
            **form_payload,
        }

    def save_config(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        tab = self.tab_config(name)
        current_data, layout, _, _ = load_yaml_with_layout(tab.yaml_path, tab.sample_candidates)
        new_data = normalize_payload_data(name, layout, payload, current_data)
        rendered = render_yaml(name, layout, new_data)
        tab.yaml_path.parent.mkdir(parents=True, exist_ok=True)
        tab.yaml_path.write_text(rendered, encoding="utf-8")
        return self.load_config(name)

    def process_snapshot(self, name: str) -> dict[str, Any]:
        self.tab_config(name)
        return self.processes[name].snapshot()

    def process_start(
        self,
        name: str,
        command: str,
        isolate_cpu: bool,
        isolate_cpu_spec: str | None = None,
        sudo_password: str = "",
    ) -> dict[str, Any]:
        tab = self.tab_config(name)
        data, _, _, _ = load_yaml_with_layout(tab.yaml_path, tab.sample_candidates)
        cpu_values = [int_or_zero(v) for v in (data.get("cpu_cores", []) or [])]
        return self.processes[name].start(command, cpu_values, isolate_cpu, isolate_cpu_spec, sudo_password)

    def process_stop(self, name: str, sudo_password: str = "") -> dict[str, Any]:
        self.tab_config(name)
        return self.processes[name].stop(sudo_password)

    def process_reset_isolation(self, name: str, sudo_password: str = "") -> dict[str, Any]:
        self.tab_config(name)
        return self.processes[name].reset_isolation(sudo_password)

    def stop_all(self) -> None:
        for controller in self.processes.values():
            controller.stop_if_running()

    def app_state_json(self) -> str:
        payload = {
            "build_dir": str(self.build_dir),
            "tabs": {
                name: {
                    "label": tab.label,
                    "default_command": tab.default_command,
                    "presets": list(tab.presets),
                }
                for name, tab in self.tabs.items()
            },
        }
        return json.dumps(payload, ensure_ascii=True)


class ConfigEditorHandler(BaseHTTPRequestHandler):
    server: "ConfigEditorServer"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._serve_index()
                return
            if parsed.path == "/api/config":
                name = self._require_name(parsed)
                self._send_json(self.server.app.load_config(name))
                return
            if parsed.path == "/api/process":
                name = self._require_name(parsed)
                self._send_json(self.server.app.process_snapshot(name))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except RuntimeError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/config/save":
                name = self._require_name_from_body(payload)
                self._send_json(self.server.app.save_config(name, payload))
                return
            if parsed.path == "/api/process/start":
                name = self._require_name_from_body(payload)
                command = payload.get("command", "")
                if not isinstance(command, str):
                    raise RuntimeError("Command must be a string.")
                isolate_cpu = payload.get("isolate_cpu", True)
                if not isinstance(isolate_cpu, bool):
                    raise RuntimeError("isolate_cpu must be a boolean.")
                isolate_cpu_spec = payload.get("isolate_cpu_spec", "")
                if not isinstance(isolate_cpu_spec, str):
                    raise RuntimeError("isolate_cpu_spec must be a string.")
                override_isolate = payload.get("override_isolate", False)
                if not isinstance(override_isolate, bool):
                    raise RuntimeError("override_isolate must be a boolean.")
                sudo_password = self._read_sudo_password(payload)
                effective_spec = isolate_cpu_spec if override_isolate else None
                self._send_json(
                    self.server.app.process_start(name, command, isolate_cpu, effective_spec, sudo_password)
                )
                return
            if parsed.path == "/api/process/stop":
                name = self._require_name_from_body(payload)
                sudo_password = self._read_sudo_password(payload)
                self._send_json(self.server.app.process_stop(name, sudo_password))
                return
            if parsed.path == "/api/process/reset-isolation":
                name = self._require_name_from_body(payload)
                sudo_password = self._read_sudo_password(payload)
                self._send_json(self.server.app.process_reset_isolation(name, sudo_password))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except RuntimeError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _serve_index(self) -> None:
        page = load_html_page().replace("{{APP_STATE_JSON}}", self.server.app.app_state_json())
        body = page.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("JSON body must be an object.")
        return payload

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _require_name(parsed: Any) -> str:
        query = parse_qs(parsed.query)
        names = query.get("name", [])
        if not names:
            raise RuntimeError("Missing 'name' query parameter.")
        return names[0]

    @staticmethod
    def _require_name_from_body(payload: dict[str, Any]) -> str:
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Missing 'name' in request body.")
        return name

    @staticmethod
    def _read_sudo_password(payload: dict[str, Any]) -> str:
        sudo_password = payload.get("sudo_password", "")
        if not isinstance(sudo_password, str):
            raise RuntimeError("sudo_password must be a string.")
        return sudo_password


class ConfigEditorServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], app: ConfigEditorApp) -> None:
        super().__init__(server_address, ConfigEditorHandler)
        self.app = app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web-based parameter/value config editor and runtime launcher for OpenISAC."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, e.g. 0.0.0.0 for remote access.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP listen port.")
    parser.add_argument(
        "--build-dir",
        default="build",
        help="Build directory that contains Modulator.yaml, Demodulator.yaml, and binaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    build_dir = (repo_root / args.build_dir).resolve()
    app = ConfigEditorApp(repo_root=repo_root, build_dir=build_dir)
    atexit.register(app.stop_all)

    server = ConfigEditorServer((args.host, args.port), app)
    print(f"OpenISAC Config Console listening on http://{args.host}:{args.port}")
    print(f"Runtime build directory: {build_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down...")
    finally:
        server.server_close()
        app.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
