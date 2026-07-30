#!/usr/bin/env python3
"""Package the local scene4 folder and update the Google Drive asset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "assets" / "scene4.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=None, help="Defaults to manifest source_dir.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to manifest cache_dir.")
    parser.add_argument("--version", default=date.today().isoformat())
    parser.add_argument("--compression-level", type=int, default=3, help="zstd level. 3 is a good speed/size tradeoff.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse an existing archive with this version.")
    parser.add_argument("--upload", action="store_true", help="Upload the archive to the manifest rclone remote.")
    parser.add_argument("--remote", default=None, help="Override the manifest rclone remote for --upload.")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "name": "scene4",
            "drive_folder_id": "1jeaOb4C-koKg0QY1ii-6CcOiWH-q6bfZ",
            "rclone_remote": "ardy-scene4",
            "cache_dir": ".cache/assets",
            "source_dir": "scene4",
            "extract_to": "scene4",
            "ready_file": "scene4/scene4/main.usd",
        }
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def package_scene(source_dir: Path, archive_path: Path, compression_level: int) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"scene4 source directory not found: {source_dir}")
    ready_file = source_dir / "scene4" / "main.usd"
    if not ready_file.exists():
        raise FileNotFoundError(f"Expected scene4 USD is missing: {ready_file}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    zstd = f"zstd -T0 -{compression_level}"
    run(
        [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            f"--use-compress-program={zstd}",
            "-cf",
            str(archive_path),
            "-C",
            str(source_dir),
            ".",
        ]
    )


def upload_archive(archive_path: Path, remote: str, remote_path: str) -> None:
    run(["rclone", "copyto", str(archive_path), f"{remote}:{remote_path}", "--progress"])


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    version = str(args.version)
    archive_name = f"scene4-v{version}.tar.zst"
    source_dir = repo_path(args.source_dir or manifest.get("source_dir", "scene4"))
    output_dir = repo_path(args.output_dir or manifest.get("cache_dir", ".cache/assets"))
    archive_path = output_dir / archive_name

    packaged = True
    if args.reuse_existing and archive_path.exists():
        print(f"Reusing archive: {archive_path}")
        packaged = False
    else:
        package_scene(source_dir, archive_path, args.compression_level)
    manifest.update(
        {
            "name": "scene4",
            "version": version,
            "archive": archive_name,
            "remote_path": archive_name,
            "cache_dir": str(output_dir.relative_to(REPO_ROOT)) if output_dir.is_relative_to(REPO_ROOT) else str(output_dir),
            "source_dir": str(source_dir.relative_to(REPO_ROOT)) if source_dir.is_relative_to(REPO_ROOT) else str(source_dir),
            "extract_to": str(manifest.get("extract_to", "scene4")),
            "ready_file": str(manifest.get("ready_file", "scene4/scene4/main.usd")),
            "sha256": sha256_file(archive_path),
            "size_bytes": archive_path.stat().st_size,
        }
    )
    write_manifest(manifest_path, manifest)
    print(f"{'Wrote' if packaged else 'Verified'} archive: {archive_path}")
    print(f"Updated manifest: {manifest_path}")

    if args.upload:
        remote = args.remote or str(manifest.get("rclone_remote", "ardy-scene4"))
        upload_archive(archive_path, remote, archive_name)


if __name__ == "__main__":
    main()
