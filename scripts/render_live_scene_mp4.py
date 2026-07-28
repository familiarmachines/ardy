#!/usr/bin/env python3
"""Render a generated ARDY motion inside a USD scene to an MP4.

This is intended for final-quality offline exports. It uses the same scene-loading,
avatar-loading, and auto-camera helpers as the live Blender server, but does not
start the HTTP server or play the live viewport.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_live_server import (  # noqa: E402
    AVATAR_OBJECT_PROP,
    WAYPOINT_OBJECT_PROP,
    configure_scene_lighting,
    import_usd,
    load_motion_into_scene,
)
from render_motion_blender import configure_frame_output, encode_video, look_at  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an ARDY live-scene motion to MP4.")
    parser.add_argument("--usd", type=Path, required=True, help="USD/USDZ/USDA scene path.")
    parser.add_argument("--motion", type=Path, required=True, help="ARDY .npz motion path.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=128, help="Eevee final render samples.")
    parser.add_argument("--raytracing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-space-reflections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shadow-size", default="2048", help="Eevee shadow map size, e.g. 1024 or 2048.")
    parser.add_argument(
        "--method",
        choices=("render", "opengl", "opengl_view"),
        default="render",
        help="Use Blender final render, camera OpenGL render, or viewport-context OpenGL render. Default: render.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("auto", "path"),
        default="auto",
        help="Use ARDY's auto camera or a static path-framing camera. Default: auto.",
    )
    parser.add_argument(
        "--viewport-shading",
        choices=("MATERIAL", "RENDERED"),
        default="RENDERED",
        help="Viewport shading used by --method opengl_view. MATERIAL is faster; RENDERED uses scene lighting.",
    )
    parser.add_argument("--frame-start", type=int, default=None, help="Optional first frame to render.")
    parser.add_argument("--frame-end", type=int, default=None, help="Optional final frame to render.")
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS. Defaults to motion FPS.")
    parser.add_argument("--render-mode", choices=("auto", "skin", "skeleton", "both"), default="skin")
    parser.add_argument("--light-intensity-scale", type=float, default=None, help="USD import light intensity scale.")
    parser.add_argument("--light-scale", type=float, default=1.0, help="Post-import light energy scale.")
    parser.add_argument("--max-light-energy", type=float, default=600.0, help="Clamp scene light energies.")
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--view-transform", default="AgX")
    parser.add_argument("--look", default=None)
    parser.add_argument("--avatar-color", type=float, nargs=3, default=(0.72, 0.55, 0.44), metavar=("R", "G", "B"))
    parser.add_argument("--world-color", type=float, nargs=3, default=(0.02, 0.02, 0.02), metavar=("R", "G", "B"))
    parser.add_argument(
        "--hide-helpers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide waypoint markers, waypoint paths, and ARDY root-path helpers during render.",
    )
    parser.add_argument(
        "--hide-scene-meshes",
        action="store_true",
        help="Diagnostic mode: hide non-avatar scene meshes during render.",
    )
    parser.add_argument(
        "--hide-name-contains",
        action="append",
        default=[],
        help="Hide objects whose name contains this case-insensitive substring; repeatable.",
    )
    parser.add_argument("--quit", action="store_true", help="Quit Blender after rendering.")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def set_if_present(obj: object, name: str, value: object) -> bool:
    if not hasattr(obj, name):
        return False
    setattr(obj, name, value)
    return True


def apply_high_quality_profile(
    width: int,
    height: int,
    samples: int,
    raytracing: bool,
    screen_space_reflections: bool,
    shadow_size: str,
) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        set_if_present(eevee, "taa_render_samples", samples)
        set_if_present(eevee, "taa_samples", min(samples, 64))
        set_if_present(eevee, "use_gtao", True)
        set_if_present(eevee, "gtao_distance", 3.0)
        set_if_present(eevee, "gtao_factor", 1.2)
        set_if_present(eevee, "use_soft_shadows", True)
        set_if_present(eevee, "shadow_cube_size", shadow_size)
        set_if_present(eevee, "shadow_cascade_size", shadow_size)
        set_if_present(eevee, "use_bloom", False)
        set_if_present(eevee, "use_ssr", screen_space_reflections)
        set_if_present(eevee, "use_raytracing", raytracing)

    view = scene.view_settings
    view.view_transform = "AgX"
    view.look = "None"
    view.exposure = 0.0
    view.gamma = 1.0


def configure_path_camera(motion_path: Path) -> bpy.types.Object:
    data = np.load(motion_path, allow_pickle=False)
    root_positions = np.asarray(data["root_positions"], dtype=np.float32)
    ground = root_positions[:, [0, 2]]
    start = ground[0]
    end = ground[-1]
    center_xy = (start + end) * 0.5
    direction_xy = end - start
    length = float(np.linalg.norm(direction_xy))
    if length < 1e-4:
        direction = Vector((0.0, 1.0, 0.0))
        length = 2.5
    else:
        direction = Vector((float(direction_xy[0] / length), float(direction_xy[1] / length), 0.0))
    side = Vector((-direction.y, direction.x, 0.0))

    center = Vector((float(center_xy[0]), float(center_xy[1]), 0.9))
    # Put the camera inside the room, beside the walking line. The wider outside
    # auto-camera can be occluded by scene4's room shell.
    camera_location = center - side * max(3.2, length * 0.6)
    camera_location.z += max(2.6, length * 0.5)

    camera = bpy.context.scene.camera
    if camera is None:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        bpy.context.scene.camera = camera
    camera.location = camera_location
    look_at(camera, center + Vector((0.0, 0.0, 0.2)))
    camera.data.lens = 16
    camera.data.dof.use_dof = False
    camera.data.clip_end = 1000.0
    bpy.context.view_layer.update()
    return camera


def set_material_input(material: bpy.types.Material, names: tuple[str, ...], value: object) -> None:
    if not material.use_nodes:
        return
    node = material.node_tree.nodes.get("Principled BSDF")
    if node is None:
        return
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return


def configure_avatar_material(color: tuple[float, float, float]) -> None:
    rgba = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.get(AVATAR_OBJECT_PROP):
            continue
        for material in obj.data.materials:
            if material is None:
                continue
            material.diffuse_color = rgba
            material.use_nodes = True
            set_material_input(material, ("Base Color",), rgba)
            set_material_input(material, ("Metallic",), 0.0)
            set_material_input(material, ("Roughness",), 0.68)


def hide_helper_objects(
    hide_scene_meshes: bool = False,
    hide_name_contains: list[str] | None = None,
) -> list[tuple[bpy.types.Object, bool, bool]]:
    hidden: list[tuple[bpy.types.Object, bool, bool]] = []
    helper_names = ("waypoint", "root_path")
    name_filters = [value.lower() for value in hide_name_contains or []]
    for obj in bpy.data.objects:
        name = obj.name.lower()
        is_helper = obj.get(WAYPOINT_OBJECT_PROP) or any(part in name for part in helper_names)
        is_scene_mesh = hide_scene_meshes and obj.type == "MESH" and not obj.get("ardy_live_avatar_object")
        is_named_hidden = any(value in name for value in name_filters)
        if is_helper or is_scene_mesh or is_named_hidden:
            hidden.append((obj, bool(obj.hide_viewport), bool(obj.hide_render)))
            obj.hide_viewport = True
            obj.hide_render = True
    return hidden


def restore_hidden_objects(hidden: list[tuple[bpy.types.Object, bool, bool]]) -> None:
    for obj, hide_viewport, hide_render in hidden:
        if obj.name in bpy.data.objects:
            obj.hide_viewport = hide_viewport
            obj.hide_render = hide_render


def apply_shading_settings(shading: bpy.types.View3DShading, shading_type: str) -> None:
    shading.type = shading_type
    if hasattr(shading, "color_type"):
        shading.color_type = "TEXTURE" if shading_type == "MATERIAL" else "MATERIAL"
    if shading_type == "RENDERED" and hasattr(shading, "use_scene_lights_render"):
        shading.use_scene_lights_render = True
    if shading_type == "RENDERED" and hasattr(shading, "use_scene_world_render"):
        shading.use_scene_world_render = True
    if shading_type == "MATERIAL" and hasattr(shading, "use_scene_lights"):
        shading.use_scene_lights = True
    if shading_type == "MATERIAL" and hasattr(shading, "use_scene_world"):
        shading.use_scene_world = True


def configure_viewport_shading(shading_type: str) -> int:
    apply_shading_settings(bpy.context.scene.display.shading, shading_type)
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
                apply_shading_settings(space.shading, shading_type)
                changed += 1
    return changed


def render_opengl_view(animation: bool, shading_type: str) -> None:
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((item for item in area.regions if item.type == "WINDOW"), None)
            space = next((item for item in area.spaces if item.type == "VIEW_3D"), None)
            if region is None or space is None:
                continue
            space.region_3d.view_perspective = "CAMERA"
            space.overlay.show_overlays = False
            with bpy.context.temp_override(window=window, area=area, region=region, space_data=space):
                bpy.ops.render.opengl(animation=animation, view_context=True)
            return
    raise RuntimeError("No VIEW_3D area is available for viewport-context OpenGL rendering.")


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    import_payload: dict[str, object] = {"path": str(args.usd.expanduser().resolve()), "clear": True}
    if args.light_intensity_scale is not None:
        import_payload["light_intensity_scale"] = args.light_intensity_scale
    import_usd(import_payload)

    lighting_payload = {
        "light_scale": args.light_scale,
        "max_energy": args.max_light_energy,
        "exposure": args.exposure,
        "gamma": args.gamma,
        "view_transform": args.view_transform,
        **({"look": args.look} if args.look is not None else {}),
        "world_color": list(args.world_color),
    }

    state = load_motion_into_scene(
        {
            "motion_path": str(args.motion.expanduser().resolve()),
            "width": args.width,
            "height": args.height,
            "render_mode": args.render_mode,
            "play": False,
            "loop": False,
            "show_root_path": False,
            "auto_camera": True,
        }
    )
    apply_high_quality_profile(
        args.width,
        args.height,
        args.samples,
        raytracing=args.raytracing,
        screen_space_reflections=args.screen_space_reflections,
        shadow_size=str(args.shadow_size),
    )
    configure_scene_lighting(lighting_payload)
    configure_avatar_material(tuple(args.avatar_color))
    if args.camera_mode == "path":
        configure_path_camera(args.motion.expanduser().resolve())
    if args.fps is not None:
        bpy.context.scene.render.fps = max(1, int(round(args.fps)))
    if args.frame_start is not None:
        bpy.context.scene.frame_start = int(args.frame_start)
    if args.frame_end is not None:
        bpy.context.scene.frame_end = int(args.frame_end)

    should_hide = args.hide_helpers or args.hide_scene_meshes or bool(args.hide_name_contains)
    hidden = (
        hide_helper_objects(args.hide_scene_meshes, args.hide_name_contains)
        if should_hide
        else []
    )
    try:
        with tempfile.TemporaryDirectory(prefix="ardy_scene_render_frames_") as tmpdir:
            configure_frame_output(Path(tmpdir))
            if args.method == "opengl_view":
                configure_viewport_shading(args.viewport_shading)
                render_opengl_view(animation=True, shading_type=args.viewport_shading)
            elif args.method == "opengl":
                configure_viewport_shading(args.viewport_shading)
                bpy.ops.render.opengl(animation=True, view_context=False)
            else:
                bpy.ops.render.render(animation=True)
            encode_video(Path(tmpdir), output, float(bpy.context.scene.render.fps))
    finally:
        restore_hidden_objects(hidden)

    camera = bpy.context.scene.camera
    print(
        f"Rendered {state['current_frame_count']} frames at "
        f"{bpy.context.scene.render.resolution_x}x{bpy.context.scene.render.resolution_y} "
        f"to {output}",
        flush=True,
    )
    if camera is not None:
        print(
            "Camera "
            f"{camera.name}: location={tuple(round(float(v), 6) for v in camera.location)} "
            f"rotation={tuple(round(float(v), 6) for v in camera.rotation_euler)}",
            flush=True,
        )
    if args.quit:
        bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
