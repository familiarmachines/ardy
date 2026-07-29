from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from forehead_camera import build_head_transform_track, transform_track  # noqa: E402


class ForeheadCameraTransformTests(unittest.TestCase):
    def test_identity_head_transform_converts_position_and_scale(self) -> None:
        joints = np.zeros((1, 2, 3), dtype=np.float32)
        joints[0, 1] = [1.0, 2.0, 3.0]
        rotations = np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (1, 2, 3, 3),
        ).copy()

        track = build_head_transform_track(
            joints,
            rotations,
            ["Hips", "Head"],
            scale=2.0,
            vertical_offset=0.5,
        )

        np.testing.assert_allclose(track.locations[0], [2.0, 6.0, 4.5])
        np.testing.assert_allclose(track.rotations[0], np.eye(3), atol=1e-6)
        self.assertEqual(track.head_index, 1)

    def test_head_forward_rotation_is_preserved_in_blender_coordinates(self) -> None:
        joints = np.zeros((1, 1, 3), dtype=np.float32)
        rotations = np.zeros((1, 1, 3, 3), dtype=np.float32)
        rotations[0, 0] = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]

        track = build_head_transform_track(joints, rotations, ["Head"])
        blender_forward = track.rotations[0] @ np.asarray([0.0, 1.0, 0.0])

        np.testing.assert_allclose(blender_forward, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            track.rotations[0] @ track.rotations[0].T,
            np.eye(3),
            atol=1e-6,
        )
        self.assertAlmostEqual(float(np.linalg.det(track.rotations[0])), 1.0)

    def test_heading_and_translation_transform_the_whole_track(self) -> None:
        joints = np.zeros((1, 1, 3), dtype=np.float32)
        joints[0, 0] = [1.0, 0.0, 0.0]
        rotations = np.eye(3, dtype=np.float32)[None, None, ...]
        track = build_head_transform_track(joints, rotations, ["Head"])

        transformed = transform_track(
            track,
            heading=np.pi / 2.0,
            translation=[0.5, 0.25, 1.0],
        )

        np.testing.assert_allclose(
            transformed.locations[0],
            [0.5, 1.25, 1.0],
            atol=1e-6,
        )
        forward = transformed.rotations[0] @ np.asarray([0.0, 1.0, 0.0])
        np.testing.assert_allclose(forward, [-1.0, 0.0, 0.0], atol=1e-6)

    def test_missing_head_joint_is_rejected(self) -> None:
        joints = np.zeros((1, 1, 3), dtype=np.float32)
        rotations = np.eye(3, dtype=np.float32)[None, None, ...]
        with self.assertRaisesRegex(ValueError, "Head joint"):
            build_head_transform_track(joints, rotations, ["Hips"])

    def test_supported_skin_assets_provide_valid_head_transforms(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "ardy/assets/skeletons/cskel27/skin_standard.npz",
            "ardy/assets/skeletons/somaskel77/skin_standard.npz",
        ):
            with self.subTest(asset=relative_path):
                with np.load(repo_root / relative_path, allow_pickle=False) as data:
                    bind = data["bind_rig_transform"]
                    track = build_head_transform_track(
                        bind[None, :, :3, 3],
                        bind[None, :, :3, :3],
                        data["rig_joint_names"].tolist(),
                    )

                self.assertEqual(track.locations.shape, (1, 3))
                self.assertEqual(track.rotations.shape, (1, 3, 3))
                self.assertAlmostEqual(
                    float(np.linalg.det(track.rotations[0])),
                    1.0,
                    places=5,
                )


if __name__ == "__main__":
    unittest.main()
