#!/usr/bin/env python3
"""Submit a prompt to a running ARDY motion API server."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ARDY motion through scripts/run_motion_api.py.")
    parser.add_argument("prompt", help="Text prompt describing the motion.")
    parser.add_argument("--url", default="http://127.0.0.1:8765/generate", help="Motion API /generate URL.")
    parser.add_argument("--model", default=None, help="Model nickname or full model name.")
    parser.add_argument("--duration", type=float, default=5.0, help="Motion duration in seconds.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output", default=None, help="Output stem/path on the server.")
    parser.add_argument("--render", action="store_true", help="Render generated .npz files to MP4 in Blender.")
    parser.add_argument(
        "--render-mode",
        choices=("auto", "skin", "skeleton", "both"),
        default=None,
        help="Blender render mode when --render is used.",
    )
    parser.add_argument("--no-postprocess", action="store_true", help="Disable motion post-processing.")
    parser.add_argument("--diffusion-steps", type=int, default=None, help="Optional denoising step count.")
    parser.add_argument("--history-frames", type=int, default=None, help="Optional history frame crop.")
    parser.add_argument(
        "--cfg-weight",
        type=float,
        nargs="+",
        default=None,
        help="One text CFG weight or two weights: text and constraint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "prompt": args.prompt,
        "duration": args.duration,
        "num_samples": args.num_samples,
        "render": args.render,
        "postprocess": not args.no_postprocess,
    }
    for key in ("model", "seed", "output", "render_mode", "diffusion_steps", "history_frames", "cfg_weight"):
        value = getattr(args, key)
        if value is not None:
            payload[key] = value

    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        sys.stderr.write(error.read().decode("utf-8") + "\n")
        raise SystemExit(error.code)

    print(json.dumps(json.loads(body), indent=2))


if __name__ == "__main__":
    main()
