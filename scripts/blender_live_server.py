#!/usr/bin/env python3
"""Persistent HTTP control server for an ARDY avatar inside a live Blender session."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import bpy
import numpy as np
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_motion_blender import (  # noqa: E402
    ANIMATED_MESHES,
    PARENTS_BY_JOINT_COUNT,
    add_camera_and_light,
    build_animation,
    build_skinned_animation,
    compute_lbs_vertices,
    configure_frame_output,
    encode_video,
    load_motion,
    load_skin_data,
    make_material,
    resolve_render_mode,
    setup_world,
    start_timeline_playback,
    stop_timeline_playback,
    update_ardy_animated_meshes,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876
AVATAR_OBJECT_PROP = "ardy_live_avatar_object"
CAMERA_OBJECT_PROP = "ardy_live_camera_object"
START_MARKER_NAME = "ardy_avatar_start"


@dataclass
class LiveBlenderState:
    usd_path: str | None = None
    avatar_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    avatar_heading: float = 0.0
    current_motion_path: str | None = None
    current_prompt: str | None = None
    current_frame_count: int = 0
    current_fps: float = 20.0
    current_render_mode: str = "auto"
    loop: bool = False
    last_render_path: str | None = None


STATE = LiveBlenderState()
TASK_QUEUE: queue.Queue["BlenderTask"] = queue.Queue()


@dataclass
class BlenderTask:
    fn: Callable[[], Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    traceback_text: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ARDY live Blender HTTP control server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind. Default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind. Default: {DEFAULT_PORT}")
    parser.add_argument("--width", type=int, default=1280, help="Default render width for loaded motions.")
    parser.add_argument("--height", type=int, default=720, help="Default render height for loaded motions.")
    parser.add_argument("--usd", type=Path, default=None, help="Optional USD scene to import at startup.")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def run_on_blender_thread(fn: Callable[[], Any], timeout: float = 300.0) -> Any:
    task = BlenderTask(fn=fn)
    TASK_QUEUE.put(task)
    if not task.event.wait(timeout):
        raise TimeoutError(f"Timed out waiting {timeout:g}s for Blender to process the request.")
    if task.error is not None:
        raise task.error
    return task.result


def process_blender_tasks() -> float:
    while True:
        try:
            task = TASK_QUEUE.get_nowait()
        except queue.Empty:
            return 0.05

        try:
            task.result = task.fn()
        except BaseException as error:  # Blender timer must keep running after request failures.
            task.error = error
            task.traceback_text = traceback.format_exc()
        finally:
            task.event.set()


def stop_playback() -> None:
    stop_timeline_playback()


def json_state() -> dict[str, Any]:
    scene = bpy.context.scene
    return {
        "status": "ok",
        "usd_path": STATE.usd_path,
        "avatar_position": STATE.avatar_position,
        "avatar_heading": STATE.avatar_heading,
        "current_motion_path": STATE.current_motion_path,
        "current_prompt": STATE.current_prompt,
        "current_frame_count": STATE.current_frame_count,
        "current_fps": STATE.current_fps,
        "current_render_mode": STATE.current_render_mode,
        "loop": STATE.loop,
        "last_render_path": STATE.last_render_path,
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "frame_current": int(scene.frame_current),
        "is_playing": bool(getattr(bpy.context.screen, "is_animation_playing", False))
        if bpy.context.screen is not None
        else False,
    }


def remove_tagged_objects(prop_name: str) -> None:
    for obj in list(bpy.data.objects):
        if obj.get(prop_name):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and getattr(data, "users", 0) == 0:
                if hasattr(bpy.data, "meshes") and data.name in bpy.data.meshes:
                    bpy.data.meshes.remove(data)
                elif hasattr(bpy.data, "curves") and data.name in bpy.data.curves:
                    bpy.data.curves.remove(data)


def clear_avatar() -> dict[str, Any]:
    stop_playback()
    ANIMATED_MESHES.clear()
    remove_tagged_objects(AVATAR_OBJECT_PROP)
    STATE.current_motion_path = None
    STATE.current_prompt = None
    STATE.current_frame_count = 0
    return json_state()


def clear_scene() -> None:
    stop_playback()
    ANIMATED_MESHES.clear()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_usd(payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(payload.get("path", ""))).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"USD scene not found: {path}")

    if bool(payload.get("clear", False)):
        clear_scene()

    bpy.ops.wm.usd_import(filepath=str(path))
    STATE.usd_path = str(path)
    return json_state()


def mark_objects(objects: list[bpy.types.Object], prop_name: str) -> None:
    for obj in objects:
        obj[prop_name] = True


def add_root_path(points: np.ndarray, material: bpy.types.Material) -> bpy.types.Object:
    curve = bpy.data.curves.new("ardy_live_root_path_curve", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = 0.012
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, coord in zip(spline.points, points):
        point.co = (float(coord[0]), float(coord[1]), 0.025, 1.0)
    obj = bpy.data.objects.new("ardy_live_root_path", curve)
    obj[AVATAR_OBJECT_PROP] = True
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj


def update_auto_camera(bounds_points: np.ndarray) -> None:
    remove_tagged_objects(CAMERA_OBJECT_PROP)
    before = set(bpy.data.objects)
    add_camera_and_light(bounds_points)
    after = set(bpy.data.objects)
    mark_objects([obj for obj in after - before], CAMERA_OBJECT_PROP)


def set_start_marker(position: list[float], heading: float) -> None:
    marker = bpy.data.objects.get(START_MARKER_NAME)
    if marker is None:
        bpy.ops.object.empty_add(type="ARROWS", location=position)
        marker = bpy.context.object
        marker.name = START_MARKER_NAME
    marker.location = position
    marker.rotation_euler = (0.0, 0.0, heading)
    marker[AVATAR_OBJECT_PROP] = True


def place_avatar(payload: dict[str, Any]) -> dict[str, Any]:
    position = payload.get("position", STATE.avatar_position)
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("position must be a JSON list [x, y, z] in Blender coordinates.")

    STATE.avatar_position = [float(value) for value in position]
    STATE.avatar_heading = float(payload.get("heading", STATE.avatar_heading))
    set_start_marker(STATE.avatar_position, STATE.avatar_heading)
    return json_state()


def load_motion_into_scene(payload: dict[str, Any]) -> dict[str, Any]:
    motion_path = Path(str(payload.get("motion_path", ""))).expanduser().resolve()
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    sample_index = int(payload.get("sample_index", 0))
    scale = float(payload.get("scale", 1.0))
    width = int(payload.get("width", bpy.context.scene.render.resolution_x or 1280))
    height = int(payload.get("height", bpy.context.scene.render.resolution_y or 720))
    render_mode = str(payload.get("render_mode", "auto"))
    vertical_offset = float(payload.get("vertical_offset", 0.0))
    play = bool(payload.get("play", True))
    loop = bool(payload.get("loop", False))
    show_root_path = bool(payload.get("show_root_path", False))
    auto_camera = bool(payload.get("auto_camera", True))

    motion = load_motion(motion_path, sample_index, scale)
    joints = motion.joints_blender.copy()
    joints[..., 2] += vertical_offset
    parents = PARENTS_BY_JOINT_COUNT.get(joints.shape[1])
    if parents is None:
        raise ValueError(f"Unsupported joint count {joints.shape[1]}.")

    skin = load_skin_data(joints.shape[1])
    resolved_mode = resolve_render_mode(render_mode, motion, skin)
    mesh_vertices = None
    bounds_points = joints
    if resolved_mode in {"skin", "both"}:
        assert skin is not None
        mesh_vertices = compute_lbs_vertices(motion, skin, scale)
        mesh_vertices[..., 2] += vertical_offset
        bounds_points = mesh_vertices

    clear_avatar()
    setup_world(bounds_points, width, height, motion.fps)

    created: list[bpy.types.Object] = []
    if resolved_mode in {"skin", "both"}:
        assert mesh_vertices is not None
        created.extend(build_skinned_animation(mesh_vertices, skin.faces))
    if resolved_mode in {"skeleton", "both"}:
        created.extend(build_animation(joints, parents, 0.035, 0.018))
    mark_objects(created, AVATAR_OBJECT_PROP)

    if show_root_path:
        created_root = add_root_path(joints[:, 0], make_material("ardy_live_root_path_material", (1.0, 0.35, 0.08, 1.0)))
        created.append(created_root)

    if auto_camera:
        update_auto_camera(bounds_points)

    STATE.current_motion_path = str(motion_path)
    STATE.current_prompt = motion.text
    STATE.current_frame_count = int(joints.shape[0])
    STATE.current_fps = float(motion.fps)
    STATE.current_render_mode = resolved_mode
    STATE.loop = loop
    update_ardy_animated_meshes(bpy.context.scene)

    save_blend = payload.get("save_blend")
    if save_blend:
        save_path = Path(str(save_blend)).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path))

    if play:
        start_timeline_playback(loop=loop)

    return json_state()


def play_scene(payload: dict[str, Any]) -> dict[str, Any]:
    loop = bool(payload.get("loop", STATE.loop))
    STATE.loop = loop
    start_timeline_playback(loop=loop)
    return json_state()


def stop_scene() -> dict[str, Any]:
    stop_playback()
    return json_state()


def render_mp4(payload: dict[str, Any]) -> dict[str, Any]:
    output = Path(str(payload.get("output", ""))).expanduser().resolve()
    if not output:
        raise ValueError("output is required.")
    output.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    with __import__("tempfile").TemporaryDirectory(prefix="ardy_live_blender_frames_") as tmpdir:
        configure_frame_output(Path(tmpdir))
        bpy.ops.render.render(animation=True)
        encode_video(Path(tmpdir), output, float(scene.render.fps))
    STATE.last_render_path = str(output)
    return json_state()


ROUTES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "/scene/load_usd": import_usd,
    "/avatar/place": place_avatar,
    "/avatar/clear": lambda _payload: clear_avatar(),
    "/motion/load": load_motion_into_scene,
    "/play": play_scene,
    "/stop": lambda _payload: stop_scene(),
    "/render/mp4": render_mp4,
}


class BlenderLiveHandler(BaseHTTPRequestHandler):
    server_version = "ARDYBlenderLive/0.1"

    def do_OPTIONS(self) -> None:
        self._send_json({"status": "ok"})

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in {"/", "/health", "/state"}:
            self._send_json(run_on_blender_thread(json_state, timeout=10.0))
            return
        self._send_json({"status": "error", "error": f"Unknown route {route}"}, status=404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        handler = ROUTES.get(route)
        if handler is None:
            self._send_json({"status": "error", "error": f"Unknown route {route}"}, status=404)
            return

        payload = self._read_json()
        try:
            self._send_json(run_on_blender_thread(lambda: handler(payload), timeout=float(payload.get("timeout", 600))))
        except Exception as error:
            traceback.print_exc()
            self._send_json(
                {
                    "status": "error",
                    "error": str(error),
                    "type": type(error).__name__,
                },
                status=500,
            )

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    args = parse_args()
    if not bpy.app.timers.is_registered(process_blender_tasks):
        bpy.app.timers.register(process_blender_tasks, first_interval=0.05)

    server = ThreadingHTTPServer((args.host, args.port), BlenderLiveHandler)
    thread = threading.Thread(target=server.serve_forever, name="ardy-blender-live-http", daemon=True)
    thread.start()
    print(f"ARDY live Blender server running at http://{args.host}:{args.port}", flush=True)

    if args.usd is not None:
        run_on_blender_thread(lambda: import_usd({"path": str(args.usd), "clear": True}), timeout=120.0)


if __name__ == "__main__":
    main()
