#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER="${BLENDER:-"$HOME/Projects/blender/blender"}"
BLENDER_PYTHON="${BLENDER_PYTHON:-"$HOME/Projects/blender/5.2/python/bin/python3.13"}"
PYTHON="${PYTHON:-python3}"
PORT="${ARDY_BLENDER_PORT:-9876}"
WIDTH="${ARDY_RENDER_WIDTH:-640}"
HEIGHT="${ARDY_RENDER_HEIGHT:-360}"
SOURCE_USD="${ARDY_SCENE4_SOURCE_USD:-"$REPO_ROOT/scene4/scene4/main.usd"}"
WRAPPER_USD="${ARDY_SCENE4_WRAPPER_USD:-"$REPO_ROOT/outputs/live_api/scene4_textured.usda"}"
FETCH_SCENE4="${ARDY_SCENE4_FETCH:-1}"
LIGHT_INTENSITY_SCALE="${ARDY_SCENE_LIGHT_INTENSITY_SCALE:-0.0001}"
MAX_LIGHT_ENERGY="${ARDY_SCENE_MAX_LIGHT_ENERGY:-600}"
EXPOSURE="${ARDY_SCENE_EXPOSURE:-0.0}"
GPU_BACKEND="${ARDY_BLENDER_GPU_BACKEND:-vulkan}"
GPU_DEVICE="${ARDY_BLENDER_GPU_DEVICE:-NVIDIA}"

if [[ ! -f "$SOURCE_USD" && "$FETCH_SCENE4" != "0" ]]; then
  "$PYTHON" "$REPO_ROOT/scripts/fetch_scene4.py"
fi

if [[ ! -f "$WRAPPER_USD" ]]; then
  if [[ ! -f "$SOURCE_USD" ]]; then
    printf 'Missing scene4 source USD: %s\n' "$SOURCE_USD" >&2
    printf 'Run: python scripts/fetch_scene4.py\n' >&2
    exit 1
  fi
  "$BLENDER_PYTHON" "$REPO_ROOT/scripts/create_usd_preview_texture_wrapper.py" "$SOURCE_USD" "$WRAPPER_USD"
fi

if [[ "$GPU_BACKEND" != "vulkan" ]]; then
  printf 'Explicit GPU selection requires the Vulkan backend, got: %s\n' "$GPU_BACKEND" >&2
  exit 1
fi

if [[ "$GPU_DEVICE" == "NVIDIA" ]]; then
  GPU_DEVICE="$(
    "$BLENDER" --gpu-backend "$GPU_BACKEND" --gpu-device help |
      awk '/NVIDIA/ { print $2; exit }'
  )"
  if [[ -z "$GPU_DEVICE" ]]; then
    printf 'No NVIDIA Vulkan device was detected by Blender.\n' >&2
    printf 'Set ARDY_BLENDER_GPU_DEVICE to a device from: blender --gpu-backend vulkan --gpu-device help\n' >&2
    exit 1
  fi
fi

printf 'Starting Blender with %s GPU device %s\n' "$GPU_BACKEND" "$GPU_DEVICE"

exec "$BLENDER" \
  --gpu-backend "$GPU_BACKEND" \
  --gpu-device "$GPU_DEVICE" \
  --gpu-device-no-fallback \
  --python "$REPO_ROOT/scripts/blender_live_server.py" -- \
  --port "$PORT" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --usd "$WRAPPER_USD" \
  --light-intensity-scale "$LIGHT_INTENSITY_SCALE" \
  --max-light-energy "$MAX_LIGHT_ENERGY" \
  --exposure "$EXPOSURE" \
  --gamma 1.0 \
  --view-transform AgX \
  --world-color 0.02 0.02 0.02 \
  --viewport-shading RENDERED \
  --use-scene-lights \
  --use-scene-world \
  --viewport-location -0.0206932481 -0.0460254699 0.0006694308 \
  --viewport-rotation 0.8370923996 0.5182799101 0.0921778455 0.1488799900 \
  --viewport-distance 5.1317672729 \
  --viewport-lens 50.0 \
  --viewport-perspective PERSP \
  --create-default-camera \
  --create-forehead-camera \
  --avatar-position 0.5 0.25 0.0 \
  --avatar-heading 0.0 \
  --show-default-avatar
