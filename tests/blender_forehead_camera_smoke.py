#!/usr/bin/env python3
"""Blender-native smoke test for the live forehead camera."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bpy  # noqa: E402

import blender_live_server as live  # noqa: E402
from forehead_camera import HeadTransformTrack  # noqa: E402


def assert_close(actual: object, expected: object, *, atol: float = 1e-5) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        atol=atol,
    )


live.STATE.forehead_camera_enabled = True
live.STATE.forehead_camera_offset = [0.0, 0.16, 0.08]
live.STATE.current_scale = 1.0
live.configure_forehead_camera()

rotations = np.asarray(
    [
        np.eye(3),
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    ],
    dtype=np.float32,
)
track = HeadTransformTrack(
    locations=np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 3.0]], dtype=np.float32),
    rotations=rotations,
    head_index=6,
)
live.set_forehead_camera_track(track)

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 2
scene.frame_set(1)
camera = bpy.data.objects[live.FOREHEAD_CAMERA_NAME]
assert_close(camera.matrix_world.translation, [1.0, 2.16, 3.08])
assert_close(
    np.asarray(camera.matrix_world.to_3x3()) @ np.asarray([0.0, 0.0, -1.0]),
    [0.0, 1.0, 0.0],
)

scene.frame_set(2)
assert_close(camera.matrix_world.translation, [1.84, 4.0, 3.08])
assert_close(
    np.asarray(camera.matrix_world.to_3x3()) @ np.asarray([0.0, 0.0, -1.0]),
    [-1.0, 0.0, 0.0],
)

default_data = bpy.data.cameras.new("test_default_camera")
default_camera = bpy.data.objects.new("test_default_camera", default_data)
scene.collection.objects.link(default_camera)
default_camera[live.DEFAULT_CAMERA_PROP] = True
live.select_camera({"camera": "forehead"})
assert scene.camera == camera
live.select_camera({"camera": "default"})
assert scene.camera == default_camera

with np.load(
    REPO_ROOT / "ardy/assets/skeletons/cskel27/skin_standard.npz",
    allow_pickle=False,
) as skin_data:
    bind = skin_data["bind_rig_transform"]
    joints = np.repeat(bind[None, :, :3, 3], 2, axis=0)
    rotations = np.repeat(bind[None, :, :3, :3], 2, axis=0)
    head_index = skin_data["rig_joint_names"].tolist().index("Head")
joints[1, head_index, 0] += 0.5

with tempfile.TemporaryDirectory(prefix="ardy_forehead_camera_test_") as tmpdir:
    motion_path = Path(tmpdir) / "motion.npz"
    np.savez(
        motion_path,
        posed_joints=joints,
        global_rot_mats=rotations,
        fps=np.asarray(20.0),
        text=np.asarray("Forehead camera smoke test"),
    )
    live.load_motion_into_scene(
        {
            "motion_path": str(motion_path),
            "render_mode": "skin",
            "play": False,
        }
    )

scene.frame_set(1)
first_motion_location = camera.matrix_world.translation.copy()
scene.frame_set(2)
second_motion_location = camera.matrix_world.translation.copy()
assert_close(second_motion_location - first_motion_location, [0.5, 0.0, 0.0])

summary = live.forehead_camera_summary()
assert summary["enabled"] is True
assert summary["exists"] is True
assert summary["tracked_frame_count"] == 2
assert summary["offset"] == [0.0, 0.16, 0.08]

live.select_camera({"camera": "forehead"})
live.remove_forehead_camera()
assert scene.camera == default_camera
assert bpy.data.objects.get(live.FOREHEAD_CAMERA_NAME) is None
assert bpy.data.objects.get(live.FOREHEAD_MOUNT_NAME) is None

print("Forehead camera Blender smoke test passed.")
