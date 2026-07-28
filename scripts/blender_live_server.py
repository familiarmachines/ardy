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
WAYPOINT_OBJECT_PROP = "ardy_live_waypoint_object"
START_MARKER_NAME = "ardy_avatar_start"
ORIGINAL_LIGHT_ENERGY_PROP = "ardy_original_light_energy"


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
    waypoints: list[dict[str, Any]] = field(default_factory=list)


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
    parser.add_argument("--light-intensity-scale", type=float, default=None, help="USD import light intensity scale.")
    parser.add_argument(
        "--no-apply-unit-conversion-scale",
        dest="apply_unit_conversion_scale",
        action="store_false",
        default=None,
        help="Disable Blender's USD unit conversion scale during import.",
    )
    parser.add_argument("--light-scale", type=float, default=None, help="Scale imported light energies after import.")
    parser.add_argument("--max-light-energy", type=float, default=None, help="Clamp imported light energies.")
    parser.add_argument("--exposure", type=float, default=None, help="Scene color-management exposure.")
    parser.add_argument("--gamma", type=float, default=None, help="Scene color-management gamma.")
    parser.add_argument("--view-transform", default=None, help='Scene color-management view transform, e.g. "AgX".')
    parser.add_argument("--look", default=None, help="Scene color-management look.")
    parser.add_argument("--world-color", type=float, nargs=3, default=None, metavar=("R", "G", "B"))
    parser.add_argument(
        "--viewport-shading",
        choices=("WIREFRAME", "SOLID", "MATERIAL", "RENDERED"),
        default=None,
        help="Set all open 3D viewports to this shading mode after startup.",
    )
    parser.add_argument("--use-scene-lights", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-scene-world", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--avatar-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional startup avatar marker position in Blender coordinates.",
    )
    parser.add_argument("--avatar-heading", type=float, default=0.0, help="Startup avatar marker heading in radians.")
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
        "waypoints": serialize_waypoints(),
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "frame_current": int(scene.frame_current),
        "is_playing": bool(getattr(bpy.context.screen, "is_animation_playing", False))
        if bpy.context.screen is not None
        else False,
    }


def scene_lighting_summary() -> dict[str, Any]:
    lights = [obj for obj in bpy.data.objects if obj.type == "LIGHT"]
    energies = [float(getattr(obj.data, "energy", 0.0)) for obj in lights]
    return {
        "count": len(lights),
        "min_energy": min(energies) if energies else None,
        "max_energy": max(energies) if energies else None,
        "avg_energy": sum(energies) / len(energies) if energies else None,
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


def cursor_location() -> list[float]:
    cursor = bpy.context.scene.cursor.location
    return [float(cursor.x), float(cursor.y), float(cursor.z)]


def serialize_waypoint(waypoint: dict[str, Any]) -> dict[str, Any]:
    serialized = {
        "index": int(waypoint["index"]),
        "label": str(waypoint["label"]),
        "position": [float(value) for value in waypoint["position"]],
    }
    for key in ("frame", "time", "heading"):
        if waypoint.get(key) is not None:
            value = waypoint[key]
            serialized[key] = int(value) if key == "frame" else float(value)
    return serialized


def serialize_waypoints() -> list[dict[str, Any]]:
    return [serialize_waypoint(waypoint) for waypoint in STATE.waypoints]


def waypoint_state(**extra: Any) -> dict[str, Any]:
    state = json_state()
    state["cursor_location"] = cursor_location()
    state["waypoint_object_count"] = sum(1 for obj in bpy.data.objects if obj.get(WAYPOINT_OBJECT_PROP))
    state.update(extra)
    return state


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
    STATE.waypoints.clear()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_usd(payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(payload.get("path", ""))).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"USD scene not found: {path}")

    if bool(payload.get("clear", False)):
        clear_scene()

    import_args: dict[str, Any] = {"filepath": str(path)}
    for key in ("light_intensity_scale", "apply_unit_conversion_scale"):
        if key in payload:
            import_args[key] = payload[key]
    bpy.ops.wm.usd_import(**import_args)
    STATE.usd_path = str(path)
    return json_state()


def configure_scene_lighting(payload: dict[str, Any]) -> dict[str, Any]:
    light_scale = float(payload.get("light_scale", payload.get("scale", 1.0)))
    max_energy = payload.get("max_energy")
    max_energy = None if max_energy is None else float(max_energy)
    exposure = payload.get("exposure")
    gamma = payload.get("gamma")
    world_color = payload.get("world_color")

    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        light = obj.data
        if ORIGINAL_LIGHT_ENERGY_PROP not in light:
            light[ORIGINAL_LIGHT_ENERGY_PROP] = float(getattr(light, "energy", 0.0))
        energy = float(light[ORIGINAL_LIGHT_ENERGY_PROP]) * light_scale
        if max_energy is not None:
            energy = min(energy, max_energy)
        light.energy = energy

    view_settings = bpy.context.scene.view_settings
    if exposure is not None:
        view_settings.exposure = float(exposure)
    if gamma is not None:
        view_settings.gamma = float(gamma)
    if "view_transform" in payload:
        view_settings.view_transform = str(payload["view_transform"])
    if "look" in payload:
        view_settings.look = str(payload["look"])

    if world_color is not None:
        if not isinstance(world_color, list) or len(world_color) != 3:
            raise ValueError("world_color must be a JSON list [r, g, b].")
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        bpy.context.scene.world.color = [float(value) for value in world_color]

    state = json_state()
    state["lighting"] = scene_lighting_summary()
    state["color_management"] = {
        "view_transform": view_settings.view_transform,
        "look": view_settings.look,
        "exposure": float(view_settings.exposure),
        "gamma": float(view_settings.gamma),
    }
    return state


def set_viewport_shading(payload: dict[str, Any]) -> dict[str, Any]:
    requested = str(payload.get("type", payload.get("mode", "MATERIAL"))).strip().upper().replace("-", "_")
    aliases = {
        "MATERIAL_PREVIEW": "MATERIAL",
        "MATERIALPREVIEW": "MATERIAL",
        "PREVIEW": "MATERIAL",
        "TEXTURED": "MATERIAL",
        "RENDER": "RENDERED",
    }
    shading_type = aliases.get(requested, requested)
    valid_types = {"WIREFRAME", "SOLID", "MATERIAL", "RENDERED"}
    if shading_type not in valid_types:
        raise ValueError(f"Unsupported viewport shading type {requested!r}; expected one of {sorted(valid_types)}.")

    use_scene_lights = bool(payload.get("use_scene_lights", shading_type == "RENDERED"))
    use_scene_world = bool(payload.get("use_scene_world", shading_type == "RENDERED"))
    changed = 0
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.shading.type = shading_type
                if hasattr(space.shading, "use_scene_lights_render"):
                    space.shading.use_scene_lights_render = use_scene_lights
                if hasattr(space.shading, "use_scene_world_render"):
                    space.shading.use_scene_world_render = use_scene_world
                changed += 1

    state = json_state()
    state["viewport_shading"] = {
        "type": shading_type,
        "viewports_changed": changed,
        "use_scene_lights": use_scene_lights,
        "use_scene_world": use_scene_world,
    }
    return state


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


def get_or_make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = make_material(name, color)
    else:
        mat.diffuse_color = color
    return mat


def normalize_waypoint_position(position: Any) -> list[float]:
    if not isinstance(position, list) or len(position) not in {2, 3}:
        raise ValueError("Waypoint position must be a JSON list [x, y] or [x, y, z].")
    x_pos = float(position[0])
    y_pos = float(position[1])
    z_pos = float(position[2]) if len(position) == 3 else 0.0
    return [x_pos, y_pos, z_pos]


def waypoint_from_payload(payload: dict[str, Any], position: list[float]) -> dict[str, Any]:
    index = len(STATE.waypoints) + 1
    label = str(payload.get("label") or f"WP{index}")
    waypoint: dict[str, Any] = {
        "index": index,
        "label": label,
        "position": position,
    }
    for key in ("frame", "time", "heading"):
        if payload.get(key) is not None:
            waypoint[key] = int(payload[key]) if key == "frame" else float(payload[key])
    return waypoint


def clear_waypoint_objects() -> None:
    remove_tagged_objects(WAYPOINT_OBJECT_PROP)


def mark_waypoint_object(obj: bpy.types.Object) -> bpy.types.Object:
    obj[WAYPOINT_OBJECT_PROP] = True
    return obj


def redraw_waypoints() -> None:
    clear_waypoint_objects()
    if not STATE.waypoints:
        return

    marker_mat = get_or_make_material("ardy_waypoint_marker_material", (0.1, 0.65, 1.0, 1.0))
    label_mat = get_or_make_material("ardy_waypoint_label_material", (1.0, 0.95, 0.35, 1.0))
    path_mat = get_or_make_material("ardy_waypoint_path_material", (1.0, 0.45, 0.05, 1.0))
    path_points: list[list[float]] = []

    for waypoint in STATE.waypoints:
        x_pos, y_pos, z_pos = [float(value) for value in waypoint["position"]]
        index = int(waypoint["index"])
        label = str(waypoint["label"])
        path_points.append([x_pos, y_pos, z_pos + 0.035])

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.08,
            depth=0.012,
            location=(x_pos, y_pos, z_pos + 0.006),
        )
        marker = mark_waypoint_object(bpy.context.object)
        marker.name = f"ardy_waypoint_{index:02d}_marker"
        marker.data.materials.append(marker_mat)

        font_curve = bpy.data.curves.new(f"ardy_waypoint_{index:02d}_label_curve", "FONT")
        font_curve.body = label
        font_curve.size = 0.12
        font_curve.align_x = "CENTER"
        font_curve.align_y = "CENTER"
        label_obj = bpy.data.objects.new(f"ardy_waypoint_{index:02d}_label", font_curve)
        label_obj.location = (x_pos, y_pos, z_pos + 0.11)
        label_obj.data.materials.append(label_mat)
        mark_waypoint_object(label_obj)
        bpy.context.collection.objects.link(label_obj)

    if len(path_points) >= 2:
        curve = bpy.data.curves.new("ardy_waypoint_path_curve", "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 2
        curve.bevel_depth = 0.015
        spline = curve.splines.new("POLY")
        spline.points.add(len(path_points) - 1)
        for point, coord in zip(spline.points, path_points):
            point.co = (coord[0], coord[1], coord[2], 1.0)
        path = bpy.data.objects.new("ardy_waypoint_path", curve)
        path.data.materials.append(path_mat)
        mark_waypoint_object(path)
        bpy.context.collection.objects.link(path)


def list_waypoints(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return waypoint_state()


def add_waypoint(payload: dict[str, Any]) -> dict[str, Any]:
    position = normalize_waypoint_position(payload.get("position"))
    waypoint = waypoint_from_payload(payload, position)
    STATE.waypoints.append(waypoint)
    redraw_waypoints()
    return waypoint_state(added_waypoint=serialize_waypoint(waypoint))


def add_waypoint_from_cursor(payload: dict[str, Any]) -> dict[str, Any]:
    cursor_payload = dict(payload)
    cursor_payload["position"] = cursor_location()
    return add_waypoint(cursor_payload)


def clear_waypoints(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    STATE.waypoints.clear()
    clear_waypoint_objects()
    return waypoint_state()


def remove_last_waypoint(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    removed = serialize_waypoint(STATE.waypoints.pop()) if STATE.waypoints else None
    for index, waypoint in enumerate(STATE.waypoints, start=1):
        waypoint["index"] = index
        waypoint["label"] = str(waypoint.get("label") or f"WP{index}")
    redraw_waypoints()
    return waypoint_state(removed_waypoint=removed)


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


def translate_location_keyframes(obj: bpy.types.Object, delta: Vector) -> bool:
    action = obj.animation_data.action if obj.animation_data is not None else None
    if action is None:
        return False

    changed = False
    for fcurve in action.fcurves:
        if fcurve.data_path != "location" or fcurve.array_index not in {0, 1, 2}:
            continue
        offset = float(delta[fcurve.array_index])
        if offset == 0.0:
            continue
        for keyframe in fcurve.keyframe_points:
            keyframe.co.y += offset
            keyframe.handle_left.y += offset
            keyframe.handle_right.y += offset
        fcurve.update()
        changed = True
    return changed


def translate_loaded_avatar(delta: Vector) -> int:
    if delta.length == 0.0:
        return 0

    moved = 0
    current_frame = int(bpy.context.scene.frame_current)
    for obj in bpy.data.objects:
        if not obj.get(AVATAR_OBJECT_PROP) or obj.name == START_MARKER_NAME:
            continue
        if translate_location_keyframes(obj, delta):
            moved += 1
        else:
            obj.location += delta
            moved += 1

    if moved:
        bpy.context.scene.frame_set(current_frame)
        update_ardy_animated_meshes(bpy.context.scene)
    return moved


def place_avatar(payload: dict[str, Any]) -> dict[str, Any]:
    position = payload.get("position", STATE.avatar_position)
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("position must be a JSON list [x, y, z] in Blender coordinates.")

    next_position = [float(value) for value in position]
    delta = Vector(next_position) - Vector(STATE.avatar_position)
    moved_objects = translate_loaded_avatar(delta)

    STATE.avatar_position = next_position
    STATE.avatar_heading = float(payload.get("heading", STATE.avatar_heading))
    set_start_marker(STATE.avatar_position, STATE.avatar_heading)
    state = json_state()
    state["moved_avatar_objects"] = moved_objects
    return state


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
    "/scene/lighting": configure_scene_lighting,
    "/viewport/shading": set_viewport_shading,
    "/avatar/place": place_avatar,
    "/avatar/clear": lambda _payload: clear_avatar(),
    "/waypoints/add": add_waypoint,
    "/waypoints/add_from_cursor": add_waypoint_from_cursor,
    "/waypoints/clear": lambda _payload: clear_waypoints(),
    "/waypoints/remove_last": lambda _payload: remove_last_waypoint(),
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
        if route == "/waypoints":
            self._send_json(run_on_blender_thread(lambda: list_waypoints(), timeout=10.0))
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
        import_payload: dict[str, Any] = {"path": str(args.usd), "clear": True}
        if args.light_intensity_scale is not None:
            import_payload["light_intensity_scale"] = args.light_intensity_scale
        if args.apply_unit_conversion_scale is not None:
            import_payload["apply_unit_conversion_scale"] = args.apply_unit_conversion_scale
        import_usd(import_payload)

    lighting_payload: dict[str, Any] = {}
    for arg_name, payload_name in (
        ("light_scale", "light_scale"),
        ("max_light_energy", "max_energy"),
        ("exposure", "exposure"),
        ("gamma", "gamma"),
        ("view_transform", "view_transform"),
        ("look", "look"),
        ("world_color", "world_color"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            lighting_payload[payload_name] = list(value) if arg_name == "world_color" else value
    if lighting_payload:
        configure_scene_lighting(lighting_payload)

    if args.viewport_shading is not None:
        shading_payload: dict[str, Any] = {"type": args.viewport_shading}
        if args.use_scene_lights is not None:
            shading_payload["use_scene_lights"] = args.use_scene_lights
        if args.use_scene_world is not None:
            shading_payload["use_scene_world"] = args.use_scene_world
        set_viewport_shading(shading_payload)

    if args.avatar_position is not None:
        place_avatar({"position": list(args.avatar_position), "heading": args.avatar_heading})


if __name__ == "__main__":
    main()
