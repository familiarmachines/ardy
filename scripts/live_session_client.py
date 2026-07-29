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
    load_usd.add_argument("--light-intensity-scale", type=float, default=None, help="Blender USD import light scale.")
    load_usd.add_argument(
        "--no-apply-unit-conversion-scale",
        dest="apply_unit_conversion_scale",
        action="store_false",
        default=None,
        help="Disable Blender's USD unit conversion scale during import.",
    )

    lighting = subparsers.add_parser("lighting", help="Configure live Blender scene lighting.")
    lighting.add_argument("--light-scale", type=float, default=None, help="Scale current light energies.")
    lighting.add_argument("--max-light-energy", type=float, default=None, help="Clamp current light energies.")
    lighting.add_argument("--exposure", type=float, default=None, help="Scene color-management exposure.")
    lighting.add_argument("--gamma", type=float, default=None, help="Scene color-management gamma.")
    lighting.add_argument("--view-transform", default=None, help='Scene view transform, e.g. "AgX".')
    lighting.add_argument("--look", default=None, help="Scene color-management look.")
    lighting.add_argument("--world-color", type=float, nargs=3, default=None, metavar=("R", "G", "B"))
    lighting.add_argument("--timeout", type=float, default=30.0)

    viewport = subparsers.add_parser("viewport-shading", help="Set the live Blender viewport shading mode.")
    viewport.add_argument("type", choices=("WIREFRAME", "SOLID", "MATERIAL", "RENDERED"))
    viewport.add_argument("--use-scene-lights", action=argparse.BooleanOptionalAction, default=None)
    viewport.add_argument("--use-scene-world", action=argparse.BooleanOptionalAction, default=None)
    viewport.add_argument("--timeout", type=float, default=30.0)

    subparsers.add_parser("waypoints", help="List waypoint markers stored in the live Blender session.")

    add_waypoint = subparsers.add_parser("add-waypoint", help="Add a waypoint from the 3D cursor or a coordinate.")
    add_waypoint.add_argument(
        "--position",
        type=float,
        nargs="+",
        default=None,
        metavar="VALUE",
        help="Explicit Blender coordinate as X Y or X Y Z. Defaults to the current Blender 3D cursor.",
    )
    add_waypoint.add_argument("--label", default=None, help="Optional marker label. Defaults to WP<N>.")
    add_waypoint.add_argument("--frame", type=int, default=None, help="Optional target frame for this waypoint.")
    add_waypoint.add_argument("--time", type=float, default=None, help="Optional target time in seconds.")
    add_waypoint.add_argument("--heading", type=float, default=None, help="Optional root heading in radians.")
    add_waypoint.add_argument("--timeout", type=float, default=30.0)

    subparsers.add_parser("clear-waypoints", help="Remove all live waypoint markers.")
    subparsers.add_parser("remove-last-waypoint", help="Remove the most recently added live waypoint marker.")

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
    prompt.add_argument(
        "--auto-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace the active scene camera with automatic motion framing.",
    )
    prompt.add_argument(
        "--no-stored-waypoints",
        action="store_true",
        help="Ignore live Blender waypoint markers when no explicit --waypoint values are provided.",
    )
    prompt.add_argument("--save-blend", default=None, help="Optional .blend path saved by Blender after loading.")
    prompt.add_argument("--render-mp4", default=None, help="Optional MP4 export path rendered by live Blender.")
    prompt.add_argument(
        "--waypoint",
        type=float,
        nargs="+",
        action="append",
        default=None,
        metavar="VALUE",
        help=(
            "Waypoint in Blender ground-plane coordinates. Use X Y for auto-spaced "
            "waypoints, FRAME X Y for explicit frame targets, or FRAME X Y HEADING; repeatable."
        ),
    )
    prompt.add_argument(
        "--waypoint-time",
        type=float,
        nargs=3,
        action="append",
        default=None,
        metavar=("SECONDS", "X", "Y"),
        help="Waypoint specified by time in seconds plus Blender ground-plane X Y; repeatable.",
    )
    prompt.add_argument(
        "--waypoints-json",
        default=None,
        help="Raw JSON waypoint list to send to the live API.",
    )

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


def build_waypoints(args: argparse.Namespace) -> list[Any] | None:
    waypoints: list[Any] = []

    if args.waypoints_json:
        try:
            raw_waypoints = json.loads(args.waypoints_json)
        except json.JSONDecodeError as error:
            raise SystemExit(f"--waypoints-json is not valid JSON: {error}") from error
        if not isinstance(raw_waypoints, list):
            raise SystemExit("--waypoints-json must decode to a JSON list.")
        waypoints.extend(raw_waypoints)

    for waypoint in args.waypoint or []:
        if len(waypoint) == 2:
            waypoints.append([float(waypoint[0]), float(waypoint[1])])
        elif len(waypoint) == 3:
            waypoints.append([int(round(waypoint[0])), float(waypoint[1]), float(waypoint[2])])
        elif len(waypoint) == 4:
            waypoints.append(
                [int(round(waypoint[0])), float(waypoint[1]), float(waypoint[2]), float(waypoint[3])]
            )
        else:
            raise SystemExit("--waypoint expects X Y, FRAME X Y, or FRAME X Y HEADING.")

    for seconds, x_pos, y_pos in args.waypoint_time or []:
        waypoints.append({"time": float(seconds), "position": [float(x_pos), float(y_pos)]})

    return waypoints or None


def build_waypoint_marker_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {"timeout": args.timeout}
    if args.position is not None:
        if len(args.position) not in {2, 3}:
            raise SystemExit("--position expects X Y or X Y Z.")
        payload["position"] = [float(value) for value in args.position]
    for key in ("label", "frame", "time", "heading"):
        value = getattr(args, key)
        if value is not None:
            payload[key] = value
    return payload


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")

    if args.command == "health":
        result = request_json("GET", f"{base_url}/health")
    elif args.command == "diagnostics":
        result = request_json("GET", f"{base_url}/diagnostics")
    elif args.command == "waypoints":
        result = request_json("GET", f"{base_url}/waypoints")
    elif args.command == "load-usd":
        payload = {"path": args.path, "clear": args.clear, "timeout": args.timeout}
        if args.light_intensity_scale is not None:
            payload["light_intensity_scale"] = args.light_intensity_scale
        if args.apply_unit_conversion_scale is not None:
            payload["apply_unit_conversion_scale"] = args.apply_unit_conversion_scale
        result = request_json(
            "POST",
            f"{base_url}/scene/load_usd",
            payload,
        )
    elif args.command == "lighting":
        payload = {"timeout": args.timeout}
        for key in ("light_scale", "max_light_energy", "exposure", "gamma", "view_transform", "look", "world_color"):
            value = getattr(args, key)
            if value is None:
                continue
            payload["max_energy" if key == "max_light_energy" else key] = list(value) if key == "world_color" else value
        result = request_json("POST", f"{base_url}/scene/lighting", payload)
    elif args.command == "viewport-shading":
        payload = {"type": args.type, "timeout": args.timeout}
        if args.use_scene_lights is not None:
            payload["use_scene_lights"] = args.use_scene_lights
        if args.use_scene_world is not None:
            payload["use_scene_world"] = args.use_scene_world
        result = request_json("POST", f"{base_url}/viewport/shading", payload)
    elif args.command == "add-waypoint":
        payload = build_waypoint_marker_payload(args)
        route = "add" if "position" in payload else "add_from_cursor"
        result = request_json("POST", f"{base_url}/waypoints/{route}", payload)
    elif args.command == "clear-waypoints":
        result = request_json("POST", f"{base_url}/waypoints/clear", {})
    elif args.command == "remove-last-waypoint":
        result = request_json("POST", f"{base_url}/waypoints/remove_last", {})
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
            "auto_camera": args.auto_camera,
            "use_stored_waypoints": not args.no_stored_waypoints,
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
        waypoints = build_waypoints(args)
        if waypoints is not None:
            payload["waypoints"] = waypoints
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
