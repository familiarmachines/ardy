#!/usr/bin/env python3
"""Render an ARDY .npz motion file to a Blender animation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


REPO_ROOT = Path(__file__).resolve().parents[1]
SKIN_ASSETS_BY_JOINT_COUNT = {
    27: REPO_ROOT / "ardy/assets/skeletons/cskel27/skin_standard.npz",
    77: REPO_ROOT / "ardy/assets/skeletons/somaskel77/skin_standard.npz",
}
ANIMATED_MESHES: list[tuple[bpy.types.Object, np.ndarray]] = []
LOOP_PLAYBACK = False
HOLD_END_PLAYBACK = False
LAST_PLAYBACK_FRAME = 0


PARENTS_BY_JOINT_COUNT = {
    27: [
        -1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16, 0, 19, 20, 21, 0,
        23, 24, 25,
    ],
    30: [
        -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22,
        23, 24, 0, 26, 27, 28,
    ],
    34: [
        -1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 0, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 17, 26, 27, 28, 29, 30, 31, 32,
    ],
    77: [
        -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17, 14, 19, 20, 21,
        22, 14, 24, 25, 26, 27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37, 3, 39, 40, 41, 42,
        43, 44, 45, 42, 47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63,
        64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75,
    ],
}


@dataclass
class MotionData:
    joints_ardy: np.ndarray
    joints_blender: np.ndarray
    global_rot_mats: np.ndarray | None
    fps: float
    text: str


@dataclass
class SkinData:
    path: Path
    bind_vertices: np.ndarray
    faces: np.ndarray
    bind_rig_transform: np.ndarray
    lbs_indices: np.ndarray
    lbs_weights: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an ARDY motion .npz with Blender.")
    parser.add_argument("motion", type=Path, help="Path to an ARDY .npz file.")
    parser.add_argument("--output", type=Path, default=None, help="Output video path. Defaults to <motion>.mp4.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index when the .npz contains a batch.")
    parser.add_argument("--width", type=int, default=1280, help="Render width.")
    parser.add_argument("--height", type=int, default=720, help="Render height.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to joint positions.")
    parser.add_argument(
        "--render-mode",
        choices=("auto", "skin", "skeleton", "both"),
        default="auto",
        help="Render a skinned mesh when available, a skeleton, or both. Default: auto.",
    )
    parser.add_argument("--joint-radius", type=float, default=0.035, help="Radius of joint markers.")
    parser.add_argument("--bone-radius", type=float, default=0.018, help="Radius of bone cylinders.")
    parser.add_argument("--save-blend", type=Path, default=None, help="Optional .blend file to save before rendering.")
    parser.add_argument("--live-only", action="store_true", help="Load the Blender scene without rendering a video.")
    parser.add_argument("--play", action="store_true", help="Start timeline playback after loading the scene.")
    parser.add_argument("--loop", action="store_true", help="Keep live playback looping over the scene frame range.")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def to_blender_points(points: np.ndarray, scale: float) -> np.ndarray:
    blender_points = np.empty_like(points, dtype=np.float32)
    blender_points[..., 0] = points[..., 0] * scale
    blender_points[..., 1] = points[..., 2] * scale
    blender_points[..., 2] = points[..., 1] * scale
    return blender_points


def select_sample(array: np.ndarray, sample_index: int, name: str, unbatched_ndim: int) -> np.ndarray:
    if array.ndim == unbatched_ndim + 1:
        if not 0 <= sample_index < array.shape[0]:
            raise IndexError(f"--sample-index {sample_index} out of range for {name} batch size {array.shape[0]}.")
        return array[sample_index]
    return array


def load_motion(path: Path, sample_index: int, scale: float) -> MotionData:
    data = np.load(path, allow_pickle=False)
    if "posed_joints" not in data:
        raise ValueError(f"{path} does not contain posed_joints.")

    joints = np.asarray(data["posed_joints"], dtype=np.float32)
    if joints.ndim == 4:
        if not 0 <= sample_index < joints.shape[0]:
            raise IndexError(f"--sample-index {sample_index} out of range for batch size {joints.shape[0]}.")
        joints = joints[sample_index]
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected posed_joints with shape [T, J, 3], got {joints.shape}.")

    global_rot_mats = None
    if "global_rot_mats" in data:
        global_rot_mats = select_sample(
            np.asarray(data["global_rot_mats"], dtype=np.float32),
            sample_index,
            "global_rot_mats",
            unbatched_ndim=4,
        )
        if global_rot_mats.shape != (*joints.shape[:2], 3, 3):
            raise ValueError(
                f"Expected global_rot_mats with shape [T, J, 3, 3] matching posed_joints, got {global_rot_mats.shape}."
            )

    fps = float(np.asarray(data["fps"]).item()) if "fps" in data else 20.0
    text = str(np.asarray(data["text"]).item()) if "text" in data else ""
    return MotionData(
        joints_ardy=joints,
        joints_blender=to_blender_points(joints, scale),
        global_rot_mats=global_rot_mats,
        fps=fps,
        text=text,
    )


def load_skin_data(joint_count: int) -> SkinData | None:
    path = SKIN_ASSETS_BY_JOINT_COUNT.get(joint_count)
    if path is None or not path.exists():
        return None

    data = np.load(path, allow_pickle=False)
    bind_rig_transform = np.asarray(data["bind_rig_transform"], dtype=np.float32)
    if bind_rig_transform.shape[0] != joint_count:
        return None

    return SkinData(
        path=path,
        bind_vertices=np.asarray(data["bind_vertices"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int64),
        bind_rig_transform=bind_rig_transform,
        lbs_indices=np.asarray(data["lbs_indices"], dtype=np.int64),
        lbs_weights=np.asarray(data["lbs_weights"], dtype=np.float32),
    )


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def clear_scene() -> None:
    ANIMATED_MESHES.clear()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def set_object_between(obj: bpy.types.Object, start: np.ndarray, end: np.ndarray, radius: float) -> None:
    a = Vector(start.tolist())
    b = Vector(end.tolist())
    delta = b - a
    length = max(delta.length, 1e-5)
    obj.location = (a + b) * 0.5
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = delta.to_track_quat("Z", "Y") if delta.length > 1e-5 else (1.0, 0.0, 0.0, 0.0)
    obj.scale = (radius, radius, length)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def set_linear_interpolation(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        if obj.animation_data is None or obj.animation_data.action is None:
            continue
        if not hasattr(obj.animation_data.action, "fcurves"):
            continue
        for fcurve in obj.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def add_root_path(points: np.ndarray, material: bpy.types.Material) -> None:
    curve = bpy.data.curves.new("root_path", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = 0.012
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, coord in zip(spline.points, points):
        point.co = (float(coord[0]), float(coord[1]), 0.025, 1.0)
    obj = bpy.data.objects.new("root_path", curve)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)


def compute_lbs_vertices(motion: MotionData, skin: SkinData, scale: float) -> np.ndarray:
    if motion.global_rot_mats is None:
        raise ValueError("Skinned rendering requires global_rot_mats in the motion .npz.")

    joints = motion.joints_ardy.astype(np.float32, copy=False)
    rotations = motion.global_rot_mats.astype(np.float32, copy=False)
    num_frames, num_joints = joints.shape[:2]
    if skin.bind_rig_transform.shape[0] != num_joints:
        raise ValueError(
            f"Skin rig has {skin.bind_rig_transform.shape[0]} joints but motion has {num_joints} joints."
        )

    transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (num_frames, num_joints, 4, 4)).copy()
    transforms[..., :3, :3] = rotations
    transforms[..., :3, 3] = joints

    bind_inv = np.linalg.inv(skin.bind_rig_transform).astype(np.float32)
    affine = (transforms @ bind_inv[None, ...])[..., :3, :]
    bind_vertices_h = np.concatenate(
        [skin.bind_vertices, np.ones((skin.bind_vertices.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    num_vertices = skin.bind_vertices.shape[0]
    max_selected_values = 18_000_000
    values_per_frame = max(1, num_vertices * skin.lbs_indices.shape[1] * 12)
    chunk_size = max(1, min(num_frames, max_selected_values // values_per_frame))
    vertices = np.empty((num_frames, num_vertices, 3), dtype=np.float32)

    for start in range(0, num_frames, chunk_size):
        end = min(num_frames, start + chunk_size)
        selected_affine = affine[start:end, skin.lbs_indices, :, :]
        transformed = np.einsum("fvwkh,vh->fvwk", selected_affine, bind_vertices_h, optimize=True)
        vertices[start:end] = (transformed * skin.lbs_weights[None, :, :, None]).sum(axis=2)

    return to_blender_points(vertices, scale)


def update_ardy_animated_meshes(scene: bpy.types.Scene) -> None:
    frame_idx = max(0, scene.frame_current - scene.frame_start)
    for obj, vertices_by_frame in ANIMATED_MESHES:
        if obj.name not in bpy.data.objects:
            continue
        clamped_idx = min(frame_idx, vertices_by_frame.shape[0] - 1)
        obj.data.vertices.foreach_set("co", vertices_by_frame[clamped_idx].reshape(-1))
        obj.data.update()


def register_mesh_frame_handler() -> None:
    handlers = bpy.app.handlers.frame_change_pre
    for handler in list(handlers):
        if getattr(handler, "__name__", "") == "update_ardy_animated_meshes":
            handlers.remove(handler)
    handlers.append(update_ardy_animated_meshes)


def build_skinned_animation(vertices_by_frame: np.ndarray, faces: np.ndarray) -> list[bpy.types.Object]:
    mesh_mat = make_material("skin_mesh_material", (0.72, 0.55, 0.44, 1.0))
    mesh = bpy.data.meshes.new("ardy_skin_mesh")
    mesh.from_pydata(vertices_by_frame[0].tolist(), [], faces.tolist())
    mesh.update()

    obj = bpy.data.objects.new("ardy_skin", mesh)
    obj.data.materials.append(mesh_mat)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)

    ANIMATED_MESHES.append((obj, vertices_by_frame))
    register_mesh_frame_handler()
    update_ardy_animated_meshes(bpy.context.scene)
    return [obj]


def setup_world(joints: np.ndarray, width: int, height: int, fps: float) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    scene.frame_start = 1
    scene.frame_end = int(joints.shape[0])
    scene.frame_set(1)
    scene.render.fps = max(1, int(round(fps)))
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.03, 0.035, 0.04)

    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 32


def add_floor(joints: np.ndarray, material: bpy.types.Material) -> None:
    flat = joints.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    span = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), 4.0)
    center = (mins + maxs) * 0.5
    bpy.ops.mesh.primitive_plane_add(size=span + 4.0, location=(float(center[0]), float(center[1]), 0.0))
    plane = bpy.context.object
    plane.name = "floor"
    plane.data.materials.append(material)


def add_camera_and_light(joints: np.ndarray) -> None:
    flat = joints.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = Vector(((mins + maxs) * 0.5).tolist())
    extent = max(float(np.linalg.norm(maxs - mins)), 2.5)

    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y - extent * 0.35, center.z + extent))
    light = bpy.context.object
    light.name = "key_light"
    light.data.energy = 650.0
    light.data.size = max(4.0, extent)

    camera_location = Vector((center.x + extent * 0.55, center.y - extent * 1.35, center.z + extent * 0.55))
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    look_at(camera, center + Vector((0.0, 0.0, 0.35)))
    camera.data.lens = 35
    camera.data.dof.use_dof = False
    bpy.context.scene.camera = camera


def build_animation(
    joints: np.ndarray,
    parents: list[int],
    joint_radius: float,
    bone_radius: float,
) -> list[bpy.types.Object]:
    joint_mat = make_material("joint_material", (0.92, 0.94, 0.97, 1.0))
    bone_mat = make_material("bone_material", (0.1, 0.45, 0.9, 1.0))
    root_mat = make_material("root_material", (1.0, 0.35, 0.08, 1.0))

    joint_objects = []
    for joint_idx in range(joints.shape[1]):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16,
            ring_count=8,
            radius=joint_radius * (1.45 if joint_idx == 0 else 1.0),
            location=joints[0, joint_idx],
        )
        obj = bpy.context.object
        obj.name = f"joint_{joint_idx:02d}"
        obj.data.materials.append(root_mat if joint_idx == 0 else joint_mat)
        joint_objects.append(obj)

    bone_objects = []
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            continue
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=1.0, depth=1.0)
        obj = bpy.context.object
        obj.name = f"bone_{parent_idx:02d}_{joint_idx:02d}"
        obj.data.materials.append(bone_mat)
        set_object_between(obj, joints[0, parent_idx], joints[0, joint_idx], bone_radius)
        bone_objects.append((obj, parent_idx, joint_idx))

    for frame_idx in range(joints.shape[0]):
        frame = frame_idx + 1
        for joint_idx, obj in enumerate(joint_objects):
            obj.location = joints[frame_idx, joint_idx]
            obj.keyframe_insert(data_path="location", frame=frame)
        for obj, parent_idx, joint_idx in bone_objects:
            set_object_between(obj, joints[frame_idx, parent_idx], joints[frame_idx, joint_idx], bone_radius)
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

    objects = joint_objects + [obj for obj, _, _ in bone_objects]
    set_linear_interpolation(objects)
    return objects


def configure_frame_output(frame_dir: Path) -> None:
    scene = bpy.context.scene
    scene.render.filepath = str(frame_dir / "frame_")
    scene.render.image_settings.file_format = "PNG"


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH; cannot encode MP4 output.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(max(1, int(round(fps)))),
        "-start_number",
        "1",
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while encoding {output_path}:\n{result.stdout}")


def keep_playback_looping() -> float | None:
    if not LOOP_PLAYBACK:
        return None

    scene = bpy.context.scene
    if scene.frame_current >= scene.frame_end:
        scene.frame_set(scene.frame_start)

    screen = bpy.context.screen
    if screen is not None and getattr(screen, "is_animation_playing", True) is False:
        try:
            bpy.ops.screen.animation_play()
        except RuntimeError:
            pass
    return 0.25


def hold_playback_at_end() -> float | None:
    global LAST_PLAYBACK_FRAME

    if not HOLD_END_PLAYBACK:
        return None

    scene = bpy.context.scene
    screen = bpy.context.screen
    current_frame = int(scene.frame_current)
    wrapped_to_start = LAST_PLAYBACK_FRAME > 0 and current_frame < LAST_PLAYBACK_FRAME
    if current_frame >= int(scene.frame_end) or wrapped_to_start:
        if screen is not None and getattr(screen, "is_animation_playing", False):
            try:
                bpy.ops.screen.animation_play()
            except RuntimeError:
                pass
        scene.frame_set(scene.frame_end)
        LAST_PLAYBACK_FRAME = int(scene.frame_end)
        return None

    LAST_PLAYBACK_FRAME = current_frame
    return 0.05


def start_timeline_playback(loop: bool = False) -> None:
    global HOLD_END_PLAYBACK, LAST_PLAYBACK_FRAME, LOOP_PLAYBACK

    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    LOOP_PLAYBACK = loop
    HOLD_END_PLAYBACK = not loop
    LAST_PLAYBACK_FRAME = int(scene.frame_start)

    if loop and not bpy.app.timers.is_registered(keep_playback_looping):
        bpy.app.timers.register(keep_playback_looping, first_interval=0.25)
    if not loop and not bpy.app.timers.is_registered(hold_playback_at_end):
        bpy.app.timers.register(hold_playback_at_end, first_interval=0.05)

    if bpy.context.screen is None:
        print("No Blender screen is available; skipping live playback.", flush=True)
        return

    try:
        bpy.ops.screen.animation_play()
    except RuntimeError as error:
        print(f"Could not start Blender timeline playback: {error}", flush=True)


def stop_timeline_playback() -> None:
    global HOLD_END_PLAYBACK, LOOP_PLAYBACK

    LOOP_PLAYBACK = False
    HOLD_END_PLAYBACK = False
    screen = bpy.context.screen
    if screen is not None and getattr(screen, "is_animation_playing", False):
        try:
            bpy.ops.screen.animation_play()
        except RuntimeError:
            pass


def resolve_render_mode(requested_mode: str, motion: MotionData, skin: SkinData | None) -> str:
    has_skin = skin is not None and motion.global_rot_mats is not None
    if requested_mode == "auto":
        return "skin" if has_skin else "skeleton"
    if requested_mode in {"skin", "both"} and not has_skin:
        reasons = []
        if skin is None:
            reasons.append(f"no skin asset for {motion.joints_ardy.shape[1]} joints")
        if motion.global_rot_mats is None:
            reasons.append("motion .npz has no global_rot_mats")
        raise ValueError(f"Cannot use --render-mode {requested_mode}: {', '.join(reasons)}.")
    return requested_mode


def render_motion_file(
    motion_path: Path,
    output_path: Path,
    sample_index: int = 0,
    width: int = 1280,
    height: int = 720,
    scale: float = 1.0,
    render_mode: str = "auto",
    joint_radius: float = 0.035,
    bone_radius: float = 0.018,
    save_blend: Path | None = None,
    live_only: bool = False,
    play: bool = False,
    loop: bool = False,
) -> None:
    motion = load_motion(motion_path, sample_index, scale)
    joints = motion.joints_blender
    parents = PARENTS_BY_JOINT_COUNT.get(joints.shape[1])
    if parents is None:
        expected = sorted(PARENTS_BY_JOINT_COUNT)
        raise ValueError(f"Unsupported joint count {joints.shape[1]}; expected one of {expected}.")

    skin = load_skin_data(joints.shape[1])
    resolved_mode = resolve_render_mode(render_mode, motion, skin)
    mesh_vertices = None
    bounds_points = joints
    if resolved_mode in {"skin", "both"}:
        assert skin is not None
        print(f"Skinning {motion_path} with {skin.path}", flush=True)
        mesh_vertices = compute_lbs_vertices(motion, skin, scale)
        bounds_points = mesh_vertices

    clear_scene()
    setup_world(bounds_points, width, height, motion.fps)

    floor_mat = make_material("floor_material", (0.18, 0.19, 0.2, 1.0))
    root_path_mat = make_material("root_path_material", (1.0, 0.35, 0.08, 1.0))
    add_floor(bounds_points, floor_mat)
    if resolved_mode in {"skin", "both"}:
        assert mesh_vertices is not None
        build_skinned_animation(mesh_vertices, skin.faces)
    if resolved_mode in {"skeleton", "both"}:
        build_animation(joints, parents, joint_radius, bone_radius)
    add_root_path(joints[:, 0], root_path_mat)
    add_camera_and_light(bounds_points)

    if motion.text:
        action = "Loading" if live_only else "Rendering"
        print(f"{action} prompt: {motion.text}", flush=True)
    print(
        f"Loaded {joints.shape[0]} frames @ {motion.fps:g} fps "
        f"(mode: {resolved_mode})",
        flush=True,
    )

    if save_blend:
        save_blend.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(save_blend.expanduser().resolve()))

    if live_only:
        if play:
            start_timeline_playback(loop)
        print("Live Blender scene is ready.", flush=True)
        return

    print(f"Rendering video to {output_path}", flush=True)
    with tempfile.TemporaryDirectory(prefix="ardy_blender_frames_") as tmpdir:
        configure_frame_output(Path(tmpdir))

        bpy.ops.render.render(animation=True)
        encode_video(Path(tmpdir), output_path, motion.fps)

    if play:
        start_timeline_playback(loop)


def main() -> None:
    args = parse_args()
    motion_path = args.motion.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else motion_path.with_suffix(".mp4")
    render_motion_file(
        motion_path=motion_path,
        output_path=output_path,
        sample_index=args.sample_index,
        width=args.width,
        height=args.height,
        scale=args.scale,
        render_mode=args.render_mode,
        joint_radius=args.joint_radius,
        bone_radius=args.bone_radius,
        save_blend=args.save_blend,
        live_only=args.live_only,
        play=args.play,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
