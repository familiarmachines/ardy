#!/usr/bin/env python3
"""Client for scripts/run_live_motion_api.py."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = "http://127.0.0.1:8766"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control an ARDY live motion session.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Live motion API base URL. Default: {DEFAULT_URL}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Show live session status.")
    subparsers.add_parser("diagnostics", help="Show ARDY and Blender diagnostics.")

    load_usd = subparsers.add_parser("load-usd", help="Import a USD scene into the persistent Blender session.")
    load_usd.add_argument("path", help="USD/USDZ/USDA scene path.")
    load_usd.add_argument("--clear", action="store_true", help="Clear the current Blender scene before import.")
    load_usd.add_argument("--timeout", type=float, default=120.0, help="Blender import timeout in seconds.")

    place = subparsers.add_parser("place", help="Set the avatar's initial Blender-space position and heading.")
    place.add_argument("--position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    place.add_argument("--heading", type=float, default=0.0, help="Heading radians; 0 faces Blender +Y.")
    place.add_argument("--keep-history", action="store_true", help="Do not reset ARDY motion history.")

    prompt = subparsers.add_parser("prompt", help="Generate a prompt and load it into the live Blender session.")
    prompt.add_argument("text", help="Motion prompt.")
    prompt.add_argument("--duration", type=float, default=5.0, help="Duration in seconds.")
    prompt.add_argument("--seed", type=int, default=None, help="Random seed.")
    prompt.add_argument("--output", default=None, help="Saved segment path/stem on the API server.")
    prompt.add_argument("--model", default=None, help="Model nickname or full model name.")
    prompt.add_argument("--diffusion-steps", type=int, default=None)
    prompt.add_argument("--history-frames", type=int, default=None)
    prompt.add_argument("--cfg-weight", type=float, nargs="+", default=None)
    prompt.add_argument("--render-mode", choices=("auto", "skin", "skeleton", "both"), default=None)
    prompt.add_argument("--render-width", type=int, default=None)
    prompt.add_argument("--render-height", type=int, default=None)
    prompt.add_argument("--no-continue", action="store_true", help="Ignore prior motion history for this prompt.")
    prompt.add_argument("--no-play", action="store_true", help="Load the motion but do not start playback.")
    prompt.add_argument("--loop", action="store_true", help="Loop this segment in Blender.")
    prompt.add_argument("--show-root-path", action="store_true")
    prompt.add_argument("--no-auto-camera", action="store_true")
    prompt.add_argument("--save-blend", default=None, help="Optional .blend path saved by Blender after loading.")
    prompt.add_argument("--render-mp4", default=None, help="Optional MP4 export path rendered by live Blender.")

    reset = subparsers.add_parser("reset", help="Clear ARDY motion history.")
    reset.add_argument("--clear-blender", action="store_true", help="Also remove the live avatar from Blender.")

    render = subparsers.add_parser("render-mp4", help="Export the current Blender animation to MP4.")
    render.add_argument("output", help="MP4 output path.")

    return parser.parse_args()


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 1200.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        sys.stderr.write(error.read().decode("utf-8") + "\n")
        raise SystemExit(error.code)


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")

    if args.command == "health":
        result = request_json("GET", f"{base_url}/health")
    elif args.command == "diagnostics":
        result = request_json("GET", f"{base_url}/diagnostics")
    elif args.command == "load-usd":
        result = request_json(
            "POST",
            f"{base_url}/scene/load_usd",
            {"path": args.path, "clear": args.clear, "timeout": args.timeout},
        )
    elif args.command == "place":
        result = request_json(
            "POST",
            f"{base_url}/avatar/place",
            {
                "position": list(args.position),
                "heading": args.heading,
                "reset_history": not args.keep_history,
            },
        )
    elif args.command == "prompt":
        payload: dict[str, Any] = {
            "prompt": args.text,
            "duration": args.duration,
            "continue": not args.no_continue,
            "play": not args.no_play,
            "loop": args.loop,
            "show_root_path": args.show_root_path,
            "auto_camera": not args.no_auto_camera,
        }
        for key in (
            "seed",
            "output",
            "model",
            "diffusion_steps",
            "history_frames",
            "cfg_weight",
            "render_mode",
            "render_width",
            "render_height",
            "save_blend",
            "render_mp4",
        ):
            value = getattr(args, key)
            if value is not None:
                payload[key] = value
        result = request_json("POST", f"{base_url}/prompt", payload)
    elif args.command == "reset":
        result = request_json("POST", f"{base_url}/reset", {"clear_blender": args.clear_blender})
    elif args.command == "render-mp4":
        result = request_json("POST", f"{base_url}/render/mp4", {"output": args.output})
    else:
        raise AssertionError(args.command)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
