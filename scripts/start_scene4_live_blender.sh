#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER="${BLENDER:-"$HOME/Projects/blender/blender"}"
BLENDER_PYTHON="${BLENDER_PYTHON:-"$HOME/Projects/blender/5.2/python/bin/python3.13"}"
PORT="${ARDY_BLENDER_PORT:-9876}"
WIDTH="${ARDY_RENDER_WIDTH:-640}"
HEIGHT="${ARDY_RENDER_HEIGHT:-360}"
SOURCE_USD="${ARDY_SCENE4_SOURCE_USD:-"$REPO_ROOT/scene4/scene4/main.usd"}"
WRAPPER_USD="${ARDY_SCENE4_WRAPPER_USD:-"$REPO_ROOT/outputs/live_api/scene4_textured.usda"}"
LIGHT_INTENSITY_SCALE="${ARDY_SCENE_LIGHT_INTENSITY_SCALE:-0.0001}"
MAX_LIGHT_ENERGY="${ARDY_SCENE_MAX_LIGHT_ENERGY:-600}"
EXPOSURE="${ARDY_SCENE_EXPOSURE:-0.0}"

if [[ ! -f "$WRAPPER_USD" ]]; then
  if [[ ! -f "$SOURCE_USD" ]]; then
    printf 'Missing scene4 source USD: %s\n' "$SOURCE_USD" >&2
    exit 1
  fi
  "$BLENDER_PYTHON" "$REPO_ROOT/scripts/create_usd_preview_texture_wrapper.py" "$SOURCE_USD" "$WRAPPER_USD"
fi

exec "$BLENDER" --python "$REPO_ROOT/scripts/blender_live_server.py" -- \
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
  --avatar-position 0.5 0.25 0.0 \
  --avatar-heading 0.0
