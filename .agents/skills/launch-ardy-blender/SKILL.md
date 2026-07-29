---
name: launch-ardy-blender
description: Launch and verify the persistent ARDY Blender development session with the textured Scene4 USD, preferred scene camera and Layout viewport, NVIDIA GPU rendering, live HTTP control server, and a visible skinned human avatar at the default position. Use when asked to start, launch, open, initialize, or reproduce the ARDY Scene4 Blender environment in a new Codex session.
---

# Launch ARDY Blender

Reuse the repository launcher so GPU, lighting, asset, and avatar defaults remain versioned with
the ARDY codebase.

## Resolve the repository

Use the first candidate containing `scripts/start_scene4_live_blender.sh`:

1. The current working directory.
2. The current Git repository root.
3. `$ARDY_REPO` when set.
4. The repository root three directories above this skill directory after resolving symlinks.

Stop with a clear error if none contains the launcher. Run all remaining commands from the resolved
repository root.

## Avoid duplicate sessions

Request `http://127.0.0.1:9876/health` before launching.

- If it responds successfully, report that Blender is already running and summarize its scene,
  avatar position, renderer, and playback state. Do not start another process or reset the scene
  unless the user explicitly requests it.
- If it does not respond, launch the server detached from the Codex worker:

  ```bash
  setsid --fork scripts/start_scene4_live_blender.sh >/tmp/ardy-blender-live.log 2>&1
  ```

  Use `/tmp/ardy-blender-live.log` to diagnose startup failures.
- If launch fails because port 9876 is occupied, identify the listener and report it. Do not kill an
  unknown process.

The launcher may fetch Scene4 when `scene4/scene4/main.usd` is absent. Surface any `rclone`
authentication error directly; do not replace or fabricate the asset.

## Wait for readiness

Poll `http://127.0.0.1:9876/health` in short intervals while retaining the persistent Blender
process. Allow the USD import to finish. Keep each individual wait below 60 seconds and continue
until health succeeds or Blender exits.

## Verify the result

Require all of the following:

- `usd_path` ends with `outputs/live_api/scene4_textured.usda`.
- `avatar_visible` is `true`.
- `avatar_position` is `[0.5, 0.25, 0.0]`.
- `gpu.backend` is `VULKAN`.
- `gpu.vendor` contains `NVIDIA`.
- `gpu.renderer` identifies the expected NVIDIA GPU.
- `viewport.workspace` is `Layout`.
- `viewport.location` is approximately `[-0.02069325, -0.04602547, 0.00066943]`.
- `viewport.rotation` is approximately `[0.8370924, 0.5182799, 0.09217785, 0.14887999]`.
- `viewport.distance` is approximately `5.1317673`, with `viewport.perspective` set to `CAMERA`.
- `camera.name` is `ardy_default_camera`.
- `camera.location` is approximately `[1.5632056, -4.3579917, 2.2882988]`.
- `camera.is_default` is `true`, and `camera.lens` is `50.0`.

The default avatar is the skinned neutral bind pose. A generated motion replaces it in the same
Blender session.

If verification fails, report the exact mismatched health fields. Do not silently accept integrated
graphics or restart a responsive Blender session.

## Report

Return the Blender API URL, Scene4 path, avatar position, active camera and viewport eye positions,
render engine, and GPU renderer. Mention whether a new process was launched or an existing healthy
process was reused.
