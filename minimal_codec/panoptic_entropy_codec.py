#!/usr/bin/env python
"""Entropy-coded trajectory bitstream for Panoptic AP-RD variants.

The codec stores exactly the object used by the AP-RD experiments: per-Gaussian
temporal keyframes for means3D. It intentionally does not store a dense NumPy
array. Gaussian ids are delta-coded, q16 keyframe positions are represented as
an absolute first keyframe plus temporal residuals, and all streams are zlib
entropy-coded.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np


MAGIC = b"APRDZ1\0"
HEADER_STRUCT = struct.Struct("<I")


def key_indices(num_frames: int, stride: int) -> np.ndarray:
    keys = list(range(0, num_frames, stride))
    if keys[-1] != num_frames - 1:
        keys.append(num_frames - 1)
    return np.array(keys, dtype=np.uint16)


def quantize_q16(values: np.ndarray, lo: np.ndarray, scale: np.ndarray) -> np.ndarray:
    q = np.rint((values.astype(np.float32) - lo[None, None, :]) / scale[None, None, :])
    return np.clip(q, 0, 65535).astype(np.uint16)


def _encode_uvarints(values: np.ndarray) -> bytes:
    out = bytearray()
    for value in np.asarray(values).reshape(-1):
        x = int(value)
        if x < 0:
            raise ValueError(f"uvarint cannot encode negative value {x}")
        while x >= 0x80:
            out.append((x & 0x7F) | 0x80)
            x >>= 7
        out.append(x)
    return bytes(out)


def _decode_uvarints(data: bytes, expected_count: int) -> np.ndarray:
    values = np.empty(expected_count, dtype=np.int64)
    value = 0
    shift = 0
    count = 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            if shift > 63:
                raise ValueError("varint is too long")
            continue
        if count >= expected_count:
            raise ValueError("decoded more varints than expected")
        values[count] = value
        count += 1
        value = 0
        shift = 0
    if shift != 0:
        raise ValueError("truncated varint stream")
    if count != expected_count:
        raise ValueError(f"decoded {count} varints, expected {expected_count}")
    return values


def _zigzag_encode(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.int64)
    return (x << 1) ^ (x >> 63)


def _zigzag_decode(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.int64)
    return ((x >> 1) ^ -(x & 1)).astype(np.int32)


def _compress(data: bytes) -> bytes:
    return zlib.compress(data, level=9)


def _decompress(data: bytes) -> bytes:
    return zlib.decompress(data)


def _write_container(out_path: Path, header: dict, section_payloads: list[tuple[str, bytes]]) -> None:
    header = dict(header)
    header["sections"] = [
        {"name": name, "codec": "zlib", "bytes": len(payload)}
        for name, payload in section_payloads
    ]
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(HEADER_STRUCT.pack(len(header_bytes)))
        f.write(header_bytes)
        for _, payload in section_payloads:
            f.write(payload)


def _read_container(path: Path) -> tuple[dict, dict[str, bytes]]:
    with path.open("rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path} is not an APRDZ bitstream")
        header_len_raw = f.read(HEADER_STRUCT.size)
        if len(header_len_raw) != HEADER_STRUCT.size:
            raise ValueError(f"{path} has a truncated header length")
        (header_len,) = HEADER_STRUCT.unpack(header_len_raw)
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path} has a truncated header")
        header = json.loads(header_bytes.decode("utf-8"))
        sections: dict[str, bytes] = {}
        for meta in header["sections"]:
            payload = f.read(int(meta["bytes"]))
            if len(payload) != int(meta["bytes"]):
                raise ValueError(f"{path} has a truncated section {meta['name']}")
            sections[str(meta["name"])] = payload
        tail = f.read(1)
        if tail:
            raise ValueError(f"{path} has trailing bytes after declared sections")
    return header, sections


def _encode_index_deltas(indices: np.ndarray) -> bytes:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return b""
    if np.any(np.diff(idx) <= 0):
        raise ValueError("Gaussian indices must be strictly increasing")
    deltas = np.empty_like(idx)
    deltas[0] = idx[0]
    deltas[1:] = idx[1:] - idx[:-1]
    return _encode_uvarints(deltas)


def _decode_index_deltas(data: bytes, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    deltas = _decode_uvarints(data, count)
    return np.cumsum(deltas, dtype=np.int64)


def pack_bitstream(
    means: np.ndarray,
    stride_map: np.ndarray,
    out_path: Path,
    *,
    label: str = "",
    source_reference: str = "",
) -> dict:
    means = np.asarray(means, dtype=np.float32)
    stride_map = np.asarray(stride_map)
    if means.ndim != 3 or means.shape[-1] != 3:
        raise ValueError(f"means must have shape T x G x 3, got {means.shape}")
    if stride_map.shape != (means.shape[1],):
        raise ValueError(f"stride_map must have shape ({means.shape[1]},), got {stride_map.shape}")

    axis_min = means.reshape(-1, 3).min(axis=0).astype(np.float32)
    axis_max = means.reshape(-1, 3).max(axis=0).astype(np.float32)
    axis_scale = np.maximum((axis_max - axis_min) / 65535.0, 1e-8).astype(np.float32)

    section_payloads: list[tuple[str, bytes]] = []
    group_rows = []
    total_keyframes = 0
    for stride in sorted(np.unique(stride_map).tolist()):
        stride_int = int(stride)
        gaussian_indices = np.flatnonzero(stride_map == stride_int).astype(np.uint32)
        keys = key_indices(means.shape[0], stride_int)
        vals = means[keys][:, gaussian_indices, :]
        q = quantize_q16(vals, axis_min, axis_scale)

        idx_raw = _encode_index_deltas(gaussian_indices)
        q0_raw = np.ascontiguousarray(q[0].astype("<u2", copy=False)).tobytes()
        dq = np.diff(q.astype(np.int32), axis=0)
        dq_raw = _encode_uvarints(_zigzag_encode(dq))

        idx_name = f"s{stride_int}_idx_delta_vbyte"
        q0_name = f"s{stride_int}_q0_u16"
        dq_name = f"s{stride_int}_dq_zzvbyte"
        idx_payload = _compress(idx_raw)
        q0_payload = _compress(q0_raw)
        dq_payload = _compress(dq_raw)
        section_payloads.extend(
            [
                (idx_name, idx_payload),
                (q0_name, q0_payload),
                (dq_name, dq_payload),
            ]
        )

        keyframes = int(len(keys) * len(gaussian_indices))
        total_keyframes += keyframes
        group_rows.append(
            {
                "stride": stride_int,
                "gaussians": int(len(gaussian_indices)),
                "keys_per_gaussian": int(len(keys)),
                "keyframes": keyframes,
                "sections": {
                    "index_deltas": idx_name,
                    "q0": q0_name,
                    "temporal_deltas": dq_name,
                },
                "raw_bytes": {
                    "index_deltas": len(idx_raw),
                    "q0": len(q0_raw),
                    "temporal_deltas": len(dq_raw),
                },
                "coded_bytes": {
                    "index_deltas": len(idx_payload),
                    "q0": len(q0_payload),
                    "temporal_deltas": len(dq_payload),
                },
            }
        )

    header = {
        "version": 1,
        "label": label,
        "source_reference": source_reference,
        "num_frames": int(means.shape[0]),
        "num_gaussians": int(means.shape[1]),
        "axis_min": axis_min.tolist(),
        "axis_scale": axis_scale.tolist(),
        "groups": group_rows,
    }
    _write_container(out_path, header, section_payloads)
    payload_bytes = int(out_path.stat().st_size)
    return {
        "payload_path": str(out_path),
        "payload_bytes": payload_bytes,
        "payload_mb": payload_bytes / 1e6,
        "total_keyframes": total_keyframes,
        "avg_keyframes_per_gaussian": total_keyframes / means.shape[1],
        "bits_per_keyframe_xyz": payload_bytes * 8.0 / max(total_keyframes, 1),
        "groups": group_rows,
        "axis_min": axis_min.tolist(),
        "axis_scale": axis_scale.tolist(),
        "bitstream_format": "APRDZ1:zlib(index-delta,varint q16 temporal residuals)",
    }


def pack_variable_key_bitstream(
    means: np.ndarray,
    key_lists: list[np.ndarray],
    out_path: Path,
    *,
    label: str = "",
    source_reference: str = "",
    method: str = "variable_keys",
    method_params: dict | None = None,
) -> dict:
    means = np.asarray(means, dtype=np.float32)
    if means.ndim != 3 or means.shape[-1] != 3:
        raise ValueError(f"means must have shape T x G x 3, got {means.shape}")
    if len(key_lists) != means.shape[1]:
        raise ValueError(f"expected {means.shape[1]} key lists, got {len(key_lists)}")

    axis_min = means.reshape(-1, 3).min(axis=0).astype(np.float32)
    axis_max = means.reshape(-1, 3).max(axis=0).astype(np.float32)
    axis_scale = np.maximum((axis_max - axis_min) / 65535.0, 1e-8).astype(np.float32)

    key_counts = np.empty(means.shape[1], dtype=np.int64)
    key_delta_chunks = []
    q0 = np.empty((means.shape[1], 3), dtype=np.uint16)
    dq_chunks = []
    total_keyframes = 0
    for gaussian_idx, keys_in in enumerate(key_lists):
        keys = np.asarray(keys_in, dtype=np.int64)
        if keys.ndim != 1 or keys.size < 2:
            raise ValueError(f"bad key list for gaussian {gaussian_idx}: {keys}")
        if keys[0] != 0 or keys[-1] != means.shape[0] - 1:
            raise ValueError(f"RDP/variable keys must include first and last frame for gaussian {gaussian_idx}")
        if np.any(np.diff(keys) <= 0):
            raise ValueError(f"keys must be strictly increasing for gaussian {gaussian_idx}")

        vals = means[keys, gaussian_idx, :]
        q = np.rint((vals - axis_min[None, :]) / axis_scale[None, :])
        q = np.clip(q, 0, 65535).astype(np.uint16)
        q0[gaussian_idx] = q[0]
        if q.shape[0] > 1:
            dq_chunks.append(np.diff(q.astype(np.int32), axis=0))
        deltas = np.empty_like(keys)
        deltas[0] = keys[0]
        deltas[1:] = keys[1:] - keys[:-1]
        key_delta_chunks.append(deltas)
        key_counts[gaussian_idx] = keys.size
        total_keyframes += int(keys.size)

    key_deltas = np.concatenate(key_delta_chunks) if key_delta_chunks else np.empty(0, dtype=np.int64)
    dq = np.concatenate(dq_chunks, axis=0) if dq_chunks else np.empty((0, 3), dtype=np.int32)

    section_payloads = [
        ("key_counts_vbyte", _compress(_encode_uvarints(key_counts))),
        ("key_deltas_vbyte", _compress(_encode_uvarints(key_deltas))),
        ("q0_u16", _compress(np.ascontiguousarray(q0.astype("<u2", copy=False)).tobytes())),
        ("dq_zzvbyte", _compress(_encode_uvarints(_zigzag_encode(dq)))),
    ]
    header = {
        "version": 1,
        "coding_mode": "variable_keys",
        "method": method,
        "method_params": method_params or {},
        "label": label,
        "source_reference": source_reference,
        "num_frames": int(means.shape[0]),
        "num_gaussians": int(means.shape[1]),
        "axis_min": axis_min.tolist(),
        "axis_scale": axis_scale.tolist(),
        "total_keyframes": int(total_keyframes),
        "avg_keyframes_per_gaussian": float(total_keyframes / means.shape[1]),
        "sections_by_name": {
            "key_counts": "key_counts_vbyte",
            "key_deltas": "key_deltas_vbyte",
            "q0": "q0_u16",
            "temporal_deltas": "dq_zzvbyte",
        },
    }
    _write_container(out_path, header, section_payloads)
    payload_bytes = int(out_path.stat().st_size)
    return {
        "payload_path": str(out_path),
        "payload_bytes": payload_bytes,
        "payload_mb": payload_bytes / 1e6,
        "total_keyframes": int(total_keyframes),
        "avg_keyframes_per_gaussian": total_keyframes / means.shape[1],
        "bits_per_keyframe_xyz": payload_bytes * 8.0 / max(total_keyframes, 1),
        "axis_min": axis_min.tolist(),
        "axis_scale": axis_scale.tolist(),
        "bitstream_format": "APRDZ1:variable-keys:zlib(key-delta,varint q16 temporal residuals)",
        "method": method,
        "method_params": method_params or {},
    }


def _decode_variable_key_bitstream(header: dict, sections: dict[str, bytes]) -> np.ndarray:
    num_frames = int(header["num_frames"])
    num_gaussians = int(header["num_gaussians"])
    axis_min = np.asarray(header["axis_min"], dtype=np.float32)
    axis_scale = np.asarray(header["axis_scale"], dtype=np.float32)
    names = header["sections_by_name"]

    key_counts = _decode_uvarints(_decompress(sections[names["key_counts"]]), num_gaussians).astype(np.int64)
    total_keyframes = int(key_counts.sum())
    key_deltas = _decode_uvarints(_decompress(sections[names["key_deltas"]]), total_keyframes).astype(np.int64)

    q0 = np.frombuffer(_decompress(sections[names["q0"]]), dtype="<u2").astype(np.int32)
    if q0.size != num_gaussians * 3:
        raise ValueError(f"bad variable-key q0 length: {q0.size} vs {num_gaussians * 3}")
    q0 = q0.reshape(num_gaussians, 3)

    expected_delta_count = max(total_keyframes - num_gaussians, 0) * 3
    dq_encoded = _decode_uvarints(_decompress(sections[names["temporal_deltas"]]), expected_delta_count)
    dq = _zigzag_decode(dq_encoded).reshape(max(total_keyframes - num_gaussians, 0), 3)

    groups: dict[tuple[int, ...], list[tuple[int, np.ndarray]]] = {}
    key_cursor = 0
    dq_cursor = 0
    for gaussian_idx, key_count in enumerate(key_counts):
        count = int(key_count)
        deltas = key_deltas[key_cursor : key_cursor + count]
        key_cursor += count
        keys = np.cumsum(deltas, dtype=np.int64)
        if count < 2 or keys[0] != 0 or keys[-1] != num_frames - 1:
            raise ValueError(f"bad decoded key range for gaussian {gaussian_idx}: {keys[:3]} ... {keys[-3:]}")

        q = np.empty((count, 3), dtype=np.int32)
        q[0] = q0[gaussian_idx]
        if count > 1:
            local_dq = dq[dq_cursor : dq_cursor + count - 1]
            dq_cursor += count - 1
            q[1:] = q[0][None, :] + np.cumsum(local_dq, axis=0)
        q = np.clip(q, 0, 65535).astype(np.float32)
        vals = q * axis_scale[None, :] + axis_min[None, :]
        groups.setdefault(tuple(int(k) for k in keys.tolist()), []).append((gaussian_idx, vals.astype(np.float32)))

    out = np.empty((num_frames, num_gaussians, 3), dtype=np.float32)
    for keys_tuple, items in groups.items():
        keys = np.asarray(keys_tuple, dtype=np.int64)
        idx = np.fromiter((item[0] for item in items), dtype=np.int64, count=len(items))
        vals = np.stack([item[1] for item in items], axis=1)
        for ka, kb, va, vb in zip(keys[:-1], keys[1:], vals[:-1], vals[1:]):
            denom = max(1, int(kb - ka))
            for t in range(int(ka), int(kb) + 1):
                alpha = (t - ka) / denom
                out[t, idx, :] = (1.0 - alpha) * va + alpha * vb
    return out


def decode_bitstream(bitstream_path: Path) -> np.ndarray:
    header, sections = _read_container(bitstream_path)
    if header.get("coding_mode") == "variable_keys":
        return _decode_variable_key_bitstream(header, sections)

    num_frames = int(header["num_frames"])
    num_gaussians = int(header["num_gaussians"])
    axis_min = np.asarray(header["axis_min"], dtype=np.float32)
    axis_scale = np.asarray(header["axis_scale"], dtype=np.float32)
    out = np.empty((num_frames, num_gaussians, 3), dtype=np.float32)

    for group in header["groups"]:
        stride = int(group["stride"])
        gaussian_count = int(group["gaussians"])
        key_count = int(group["keys_per_gaussian"])
        names = group["sections"]
        idx = _decode_index_deltas(
            _decompress(sections[names["index_deltas"]]),
            gaussian_count,
        )
        q0 = np.frombuffer(_decompress(sections[names["q0"]]), dtype="<u2").astype(np.int32)
        expected_q0 = gaussian_count * 3
        if q0.size != expected_q0:
            raise ValueError(f"bad q0 length for stride {stride}: {q0.size} vs {expected_q0}")
        q0 = q0.reshape(gaussian_count, 3)

        expected_delta_count = max(key_count - 1, 0) * gaussian_count * 3
        dq_encoded = _decode_uvarints(
            _decompress(sections[names["temporal_deltas"]]),
            expected_delta_count,
        )
        dq = _zigzag_decode(dq_encoded).reshape(max(key_count - 1, 0), gaussian_count, 3)
        q = np.empty((key_count, gaussian_count, 3), dtype=np.int32)
        q[0] = q0
        if key_count > 1:
            q[1:] = q0[None, :, :] + np.cumsum(dq, axis=0)
        q = np.clip(q, 0, 65535).astype(np.float32)
        vals = q * axis_scale[None, None, :] + axis_min[None, None, :]

        keys = key_indices(num_frames, stride).astype(np.int64)
        if len(keys) != key_count:
            raise ValueError(f"key count mismatch for stride {stride}: {len(keys)} vs {key_count}")
        for ka, kb, va, vb in zip(keys[:-1], keys[1:], vals[:-1], vals[1:]):
            denom = max(1, int(kb - ka))
            for t in range(int(ka), int(kb) + 1):
                alpha = (t - ka) / denom
                out[t, idx, :] = (1.0 - alpha) * va + alpha * vb
    return out


def inspect_bitstream(bitstream_path: Path) -> dict:
    header, _ = _read_container(bitstream_path)
    return header
