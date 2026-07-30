#!/usr/bin/env python3
"""Fetch the private scene4 asset pack from Google Drive via rclone."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "assets" / "scene4.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--remote", default=None, help="Override the rclone remote from the manifest.")
    parser.add_argument("--archive", type=Path, default=None, help="Use an existing local archive instead of rclone.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override the local archive cache directory.")
    parser.add_argument("--force-download", action="store_true", help="Download even when a verified archive exists.")
    parser.add_argument("--force-extract", action="store_true", help="Extract even when the ready file already exists.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the extract directory before extraction. Use only when replacing the local scene4 asset.",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip size and SHA256 verification.")
    parser.add_argument("--print-config", action="store_true", help="Print expected rclone setup and exit.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = ("name", "archive", "remote_path", "extract_to", "ready_file")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"{manifest_path} is missing required keys: {', '.join(missing)}")
    return manifest


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, manifest: dict[str, Any], *, skip_verify: bool = False) -> bool:
    if skip_verify:
        return True

    expected_size = manifest.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        print(
            f"Archive size mismatch for {path}: got {path.stat().st_size}, expected {expected_size}.",
            file=sys.stderr,
        )
        return False

    expected_sha = manifest.get("sha256")
    if not expected_sha:
        print(
            "Manifest does not contain sha256. Rebuild it with scripts/package_scene4.py "
            "or rerun with --skip-verify.",
            file=sys.stderr,
        )
        return False
    actual_sha = sha256_file(path)
    if actual_sha != str(expected_sha).lower():
        print(
            f"Archive SHA256 mismatch for {path}: got {actual_sha}, expected {expected_sha}.",
            file=sys.stderr,
        )
        return False
    return True


def rclone_remote_exists(remote: str) -> bool:
    result = subprocess.run(
        ["rclone", "listremotes"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return f"{remote}:" in {line.strip() for line in result.stdout.splitlines()}


def print_config_help(manifest: dict[str, Any], remote: str) -> None:
    folder_id = manifest.get("drive_folder_id", "")
    print(
        "\n".join(
            [
                "Expected one-time rclone setup:",
                "",
                "  rclone config",
                "",
                f"Create a Google Drive remote named: {remote}",
                f"Set advanced option root_folder_id to: {folder_id}",
                "",
                "Then verify access with:",
                "",
                f"  rclone lsf {remote}:",
            ]
        )
    )


def ensure_rclone(remote: str, manifest: dict[str, Any]) -> None:
    if shutil.which("rclone") is None:
        print(
            "\n".join(
                [
                    "rclone is not installed.",
                    "",
                    "Install on Ubuntu/Linux with:",
                    "",
                    "  curl https://rclone.org/install.sh | sudo bash",
                    "",
                ]
            ),
            file=sys.stderr,
        )
        print_config_help(manifest, remote)
        raise SystemExit(2)
    if not rclone_remote_exists(remote):
        print(f"rclone remote {remote!r} is not configured.", file=sys.stderr)
        print_config_help(manifest, remote)
        raise SystemExit(2)


def download_archive(manifest: dict[str, Any], archive_path: Path, remote: str) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    remote_path = str(manifest["remote_path"]).lstrip("/")
    run(["rclone", "copyto", f"{remote}:{remote_path}", str(archive_path), "--progress"])


def extract_archive(archive_path: Path, extract_dir: Path, *, clean: bool) -> None:
    if clean and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    run(["tar", "--zstd", "-xf", str(archive_path), "-C", str(extract_dir)])


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    remote = args.remote or str(manifest.get("rclone_remote", "ardy-scene4"))

    if args.print_config:
        print_config_help(manifest, remote)
        return

    ready_file = repo_path(str(manifest["ready_file"]))
    extract_dir = repo_path(str(manifest["extract_to"]))
    cache_dir = repo_path(args.cache_dir or manifest.get("cache_dir", ".cache/assets"))
    archive_path = args.archive.expanduser().resolve() if args.archive else cache_dir / str(manifest["archive"])

    if ready_file.exists() and not args.force_extract:
        print(f"scene4 is already available: {ready_file}")
        return

    if args.archive is None:
        archive_valid = archive_path.exists() and verify_archive(
            archive_path,
            manifest,
            skip_verify=args.skip_verify,
        )
        if args.force_download or not archive_valid:
            ensure_rclone(remote, manifest)
            download_archive(manifest, archive_path, remote)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if not verify_archive(archive_path, manifest, skip_verify=args.skip_verify):
        raise SystemExit(1)

    extract_archive(archive_path, extract_dir, clean=args.clean)
    if not ready_file.exists():
        raise FileNotFoundError(f"Extraction finished, but expected ready file is missing: {ready_file}")
    print(f"scene4 ready: {ready_file}")


if __name__ == "__main__":
    main()
