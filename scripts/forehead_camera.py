#!/usr/bin/env python3
"""Coordinate transforms for an ARDY head-mounted Blender camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


ARDY_TO_BLENDER_BASIS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class HeadTransformTrack:
    locations: np.ndarray
    rotations: np.ndarray
    head_index: int


def build_head_transform_track(
    joints_ardy: np.ndarray,
    global_rot_mats: np.ndarray,
    joint_names: Sequence[str],
    *,
    scale: float = 1.0,
    vertical_offset: float = 0.0,
) -> HeadTransformTrack:
    """Convert ARDY Head transforms into Blender world-space transforms."""
    joints = np.asarray(joints_ardy, dtype=np.float32)
    rotations = np.asarray(global_rot_mats, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints with shape [T, J, 3], got {joints.shape}.")
    if rotations.shape != (*joints.shape[:2], 3, 3):
        raise ValueError(
            "Expected global rotations with shape [T, J, 3, 3] matching joints, "
            f"got {rotations.shape}."
        )
    if len(joint_names) != joints.shape[1]:
        raise ValueError(
            f"Expected {joints.shape[1]} joint names, got {len(joint_names)}."
        )

    try:
        head_index = list(joint_names).index("Head")
    except ValueError as error:
        raise ValueError("The motion skeleton does not contain a Head joint.") from error

    basis = ARDY_TO_BLENDER_BASIS
    locations = np.einsum(
        "ij,tj->ti",
        basis,
        joints[:, head_index],
        optimize=True,
    )
    locations *= float(scale)
    locations[:, 2] += float(vertical_offset)

    head_rotations = rotations[:, head_index]
    blender_rotations = np.einsum(
        "ij,tjk,kl->til",
        basis,
        head_rotations,
        basis.T,
        optimize=True,
    )
    return HeadTransformTrack(
        locations=np.asarray(locations, dtype=np.float32),
        rotations=np.asarray(blender_rotations, dtype=np.float32),
        head_index=head_index,
    )


def transform_track(
    track: HeadTransformTrack,
    *,
    heading: float = 0.0,
    translation: Sequence[float] = (0.0, 0.0, 0.0),
) -> HeadTransformTrack:
    """Apply a Blender Z heading and world-space translation to a track."""
    cosine = float(np.cos(heading))
    sine = float(np.sin(heading))
    heading_rotation = np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    offset = np.asarray(translation, dtype=np.float32)
    if offset.shape != (3,):
        raise ValueError(f"Expected translation [x, y, z], got shape {offset.shape}.")

    return HeadTransformTrack(
        locations=np.einsum(
            "ij,tj->ti",
            heading_rotation,
            track.locations,
            optimize=True,
        )
        + offset,
        rotations=np.einsum(
            "ij,tjk->tik",
            heading_rotation,
            track.rotations,
            optimize=True,
        ),
        head_index=track.head_index,
    )
