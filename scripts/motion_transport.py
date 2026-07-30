"""Portable transport for generated ARDY motion files."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
import tempfile
from pathlib import Path
from typing import Any

MOTION_TRANSFER_VERSION = 1
MAX_MOTION_BYTES = 64 * 1024 * 1024
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def encode_motion_file(path: Path) -> dict[str, Any]:
    """Encode a motion file for transport to a Blender process on another host."""
    resolved = path.expanduser().resolve()
    data = resolved.read_bytes()
    if len(data) > MAX_MOTION_BYTES:
        raise ValueError(f"Motion file is {len(data)} bytes; maximum transport size is {MAX_MOTION_BYTES} bytes.")

    return {
        "version": MOTION_TRANSFER_VERSION,
        "name": resolved.name,
        "encoding": "base64",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "data": base64.b64encode(data).decode("ascii"),
    }


def materialize_motion_file(payload: dict[str, Any], output_dir: Path) -> Path:
    """Validate an encoded motion payload and write it to a local cache atomically."""
    if not isinstance(payload, dict):
        raise TypeError("motion_file must be a JSON object.")
    if payload.get("version") != MOTION_TRANSFER_VERSION:
        raise ValueError(f"Unsupported motion transfer version: {payload.get('version')!r}.")
    if payload.get("encoding") != "base64":
        raise ValueError(f"Unsupported motion transfer encoding: {payload.get('encoding')!r}.")

    name = _safe_motion_name(payload.get("name"))
    declared_size = payload.get("size")
    if not isinstance(declared_size, int) or isinstance(declared_size, bool) or declared_size < 0:
        raise ValueError("motion_file.size must be a non-negative integer.")
    if declared_size > MAX_MOTION_BYTES:
        raise ValueError(f"Motion payload declares {declared_size} bytes; maximum is {MAX_MOTION_BYTES} bytes.")

    encoded = payload.get("data")
    if not isinstance(encoded, str):
        raise TypeError("motion_file.data must be a base64 string.")
    max_encoded_length = 4 * ((MAX_MOTION_BYTES + 2) // 3)
    if len(encoded) > max_encoded_length:
        raise ValueError("Encoded motion payload exceeds the maximum transport size.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("motion_file.data is not valid base64.") from error

    if len(data) != declared_size:
        raise ValueError(f"Motion payload size mismatch: expected {declared_size} bytes, received {len(data)}.")

    expected_digest = payload.get("sha256")
    actual_digest = hashlib.sha256(data).hexdigest()
    if not isinstance(expected_digest, str) or not hmac.compare_digest(expected_digest.lower(), actual_digest):
        raise ValueError("Motion payload SHA-256 checksum mismatch.")

    destination_dir = output_dir.expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{actual_digest[:16]}_{name}"
    if destination.exists() and _sha256_file(destination) == actual_digest:
        return destination

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".ardy-motion-",
            suffix=".tmp",
            dir=destination_dir,
            delete=False,
        ) as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


def _safe_motion_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("motion_file.name must be a non-empty string.")
    name = _SAFE_NAME_PATTERN.sub("_", Path(value).name).strip("._")
    if not name:
        name = "motion.npz"
    if not name.lower().endswith(".npz"):
        raise ValueError("motion_file.name must end with .npz.")
    return name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
