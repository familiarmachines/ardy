#!/usr/bin/env python3
"""Stateful ARDY prompt API that drives a persistent live Blender session."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import torch

from ardy.constraints import Root2DConstraintSet
from ardy.model import DEFAULT_MODEL
from ardy.model.loading import get_env_var
from ardy.skeleton import SOMASkeleton30
from ardy.tools import seed_everything, to_numpy

from run_motion_api import (  # noqa: E402
    APIConfig,
    MotionGenerator,
    SetupError,
    default_device,
    humanize_setup_error,
    parse_cfg_weight,
    resolve_blender,
    resolve_output_base,
    save_motion_npz,
    select_sample,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_BLENDER_URL = "http://127.0.0.1:9876"


@dataclass
class LiveAPIConfig:
    host: str
    port: int
    default_model: str
    device: str
    checkpoints_dir: str | None
    output_dir: Path
    blender_url: str
    render_width: int
    render_height: int
    render_mode: str
    text_encoder_mode: str | None
    text_encoder_url: str | None
    text_encoder_fp32: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a stateful ARDY API for a persistent live Blender session.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind. Default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind. Default: {DEFAULT_PORT}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Default ARDY model nickname or full model name.")
    parser.add_argument("--device", default=None, help='Device, e.g. "cuda:0" or "cpu". Default: cuda:0 if available.')
    parser.add_argument("--checkpoints-dir", default=None, help="Local checkpoint directory. Defaults to CHECKPOINTS_DIR.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/live_api"), help="Directory for saved segments.")
    parser.add_argument("--blender-url", default=DEFAULT_BLENDER_URL, help="Live Blender server base URL.")
    parser.add_argument("--render-width", type=int, default=1280, help="Default Blender viewport/render width.")
    parser.add_argument("--render-height", type=int, default=720, help="Default Blender viewport/render height.")
    parser.add_argument(
        "--render-mode",
        choices=("auto", "skin", "skeleton", "both"),
        default="auto",
        help="Default Blender render mode. Default: auto.",
    )
    parser.add_argument("--text-encoder-mode", choices=("auto", "api", "local"), default=None)
    parser.add_argument("--text-encoder-url", default=None)
    parser.add_argument("--text-encoder-fp32", action="store_true")
    parser.add_argument("--lazy-load", action="store_true", help="Start without loading the default model first.")
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        raise RuntimeError(f"{url} failed with HTTP {error.code}: {body}") from error


def get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def default_segment_name(prompt: str, index: int) -> str:
    words = [part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in prompt).split() if part][:6]
    slug = "_".join(words) or "motion"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{index:04d}_{slug}"


def blender_position_to_ardy_translation(position: list[float]) -> torch.Tensor:
    # ARDY is Y-up with X/Z on the ground plane. Blender is Z-up with X/Y on the ground plane.
    return torch.tensor([[float(position[0]), 0.0, float(position[1])]], dtype=torch.float32)


def blender_ground_to_ardy_root_2d(position: list[float] | tuple[float, ...]) -> list[float]:
    if len(position) < 2:
        raise ValueError("Waypoint position must contain at least [x, y] in Blender ground-plane coordinates.")
    return [float(position[0]), float(position[1])]


def ardy_root_2d_to_blender_ground(position: torch.Tensor | np.ndarray | list[float]) -> list[float]:
    values = position.detach().cpu().numpy() if isinstance(position, torch.Tensor) else np.asarray(position)
    return [float(values[0]), float(values[1])]


class LiveMotionSession:
    def __init__(self, config: LiveAPIConfig):
        self.config = config
        self._lock = threading.Lock()
        generator_config = APIConfig(
            host=config.host,
            port=config.port,
            default_model=config.default_model,
            device=config.device,
            checkpoints_dir=config.checkpoints_dir,
            output_dir=config.output_dir,
            blender=resolve_blender(None),
            render_by_default=False,
            render_width=config.render_width,
            render_height=config.render_height,
            render_mode=config.render_mode,
            text_encoder_mode=config.text_encoder_mode,
            text_encoder_url=config.text_encoder_url,
            text_encoder_fp32=config.text_encoder_fp32,
        )
        self.generator = MotionGenerator(generator_config)
        self.motion_tensor: torch.Tensor | None = None
        self.segment_index = 0
        self.avatar_position_blender = [0.0, 0.0, 0.0]
        self.init_heading = 0.0
        self.max_stored_frames = 4000
        self.last_prompt: str | None = None
        self.last_motion_path: str | None = None
        self.last_waypoints: list[dict[str, Any]] = []

    def preload(self) -> None:
        self.generator.preload()

    def health(self) -> dict[str, Any]:
        blender = self._try_blender_health()
        return {
            "status": "ok",
            "device": self.config.device,
            "default_model": self.config.default_model,
            "loaded_models": sorted(self.generator._models),
            "blender_url": self.config.blender_url,
            "blender": blender,
            "history_frames": int(self.motion_tensor.shape[1]) if self.motion_tensor is not None else 0,
            "avatar_position": self.avatar_position_blender,
            "init_heading": self.init_heading,
            "last_prompt": self.last_prompt,
            "last_motion_path": self.last_motion_path,
            "last_waypoints": self.last_waypoints,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "generator": self.generator.diagnostics(),
            "blender": self._try_blender_health(),
        }

    def _try_blender_health(self) -> dict[str, Any]:
        try:
            return get_json(urljoin(self.config.blender_url + "/", "health"), timeout=2.0)
        except Exception as error:
            return {"status": "error", "error": str(error), "type": type(error).__name__}

    def _post_blender(self, route: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
        return post_json(urljoin(self.config.blender_url + "/", route.lstrip("/")), payload, timeout=timeout)

    def load_usd(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_blender("/scene/load_usd", payload, timeout=float(payload.get("timeout", 120.0)))

    def place_avatar(self, payload: dict[str, Any]) -> dict[str, Any]:
        position = payload.get("position", self.avatar_position_blender)
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("position must be a JSON list [x, y, z] in Blender coordinates.")

        with self._lock:
            self.avatar_position_blender = [float(value) for value in position]
            self.init_heading = float(payload.get("heading", self.init_heading))
            if bool(payload.get("reset_history", True)):
                self.motion_tensor = None
            blender = self._post_blender(
                "/avatar/place",
                {"position": self.avatar_position_blender, "heading": self.init_heading},
                timeout=30.0,
            )
            return {
                "status": "ok",
                "avatar_position": self.avatar_position_blender,
                "init_heading": self.init_heading,
                "history_frames": int(self.motion_tensor.shape[1]) if self.motion_tensor is not None else 0,
                "blender": blender,
            }

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.motion_tensor = None
            self.segment_index = 0
            self.last_prompt = None
            self.last_motion_path = None
            self.last_waypoints = []
            blender = None
            if bool(payload.get("clear_blender", False)):
                blender = self._post_blender("/avatar/clear", {}, timeout=30.0)
            return {"status": "ok", "history_frames": 0, "blender": blender}

    def prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Request JSON must include a non-empty 'prompt'.")

        with self._lock:
            start = time.time()
            model_name = str(payload.get("model") or self.config.default_model)
            model = self.generator._load_model(model_name)
            if payload.get("seed") is not None:
                seed_everything(int(payload["seed"]))

            fps = float(model.motion_rep.fps)
            duration = float(payload.get("duration", 5.0))
            if duration <= 0:
                raise ValueError("duration must be positive.")
            requested_new_frames = max(1, int(round(duration * fps)))

            diffusion_steps = self.generator._resolve_diffusion_steps(model, payload.get("diffusion_steps"))
            cfg_weight = parse_cfg_weight(payload.get("cfg_weight"))
            continue_from_history = bool(payload.get("continue", True))
            history_frames = self.generator._resolve_history_frames(model, payload.get("history_frames"))
            text_feat, text_pad_mask = model._encode_text([prompt])
            waypoints = self._parse_waypoints(payload.get("waypoints"), requested_new_frames, fps)

            previous_end_motion = (
                self.motion_tensor[:, -1:].detach().clone()
                if continue_from_history and self.motion_tensor is not None and self.motion_tensor.shape[1] > 0
                else None
            )

            new_windows = self._generate_windows(
                model=model,
                prompt=prompt,
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                requested_new_frames=requested_new_frames,
                diffusion_steps=diffusion_steps,
                cfg_weight=cfg_weight,
                history_frames=history_frames,
                continue_from_history=continue_from_history,
                waypoints=waypoints,
            )

            if previous_end_motion is not None:
                visible_motion = torch.cat([previous_end_motion] + new_windows, dim=1)
            else:
                visible_motion = torch.cat(new_windows, dim=1)

            visible_output = model.motion_rep.inverse(visible_motion, is_normalized=True)
            if isinstance(model.skeleton, SOMASkeleton30):
                visible_output = model.skeleton.output_to_SOMASkeleton77(visible_output)
            visible_output_np = to_numpy(visible_output)

            output_base = resolve_output_base(
                payload.get("output") or default_segment_name(prompt, self.segment_index),
                prompt,
                self.config.output_dir,
            ).resolve()
            motion_path = output_base if output_base.suffix == ".npz" else output_base.with_suffix(".npz")
            motion_path.parent.mkdir(parents=True, exist_ok=True)
            save_motion_npz(motion_path, select_sample(visible_output_np, 0, 1), fps, prompt)

            vertical_offset = float(self.avatar_position_blender[2])
            blender = self._post_blender(
                "/motion/load",
                {
                    "motion_path": str(motion_path),
                    "width": int(payload.get("render_width", self.config.render_width)),
                    "height": int(payload.get("render_height", self.config.render_height)),
                    "render_mode": str(payload.get("render_mode", self.config.render_mode)),
                    "vertical_offset": vertical_offset,
                    "play": bool(payload.get("play", True)),
                    "loop": bool(payload.get("loop", False)),
                    "show_root_path": bool(payload.get("show_root_path", False)),
                    "auto_camera": bool(payload.get("auto_camera", True)),
                    "save_blend": payload.get("save_blend"),
                },
                timeout=float(payload.get("blender_timeout", 600.0)),
            )

            render_response = None
            render_output = payload.get("render_mp4")
            if render_output:
                render_response = self._post_blender("/render/mp4", {"output": str(render_output)}, timeout=1200.0)

            continuity_delta = None
            if previous_end_motion is not None:
                prev_output = model.motion_rep.inverse(previous_end_motion, is_normalized=True)
                prev_root = prev_output["root_positions"][0, -1].detach().cpu().numpy()
                visible_root = visible_output["root_positions"][0, 0].detach().cpu().numpy()
                continuity_delta = float(np.linalg.norm(visible_root - prev_root))

            waypoint_errors = self._measure_waypoint_errors(
                visible_output=visible_output,
                waypoints=waypoints,
                visible_frame_offset=1 if previous_end_motion is not None else 0,
            )

            self.segment_index += 1
            self.last_prompt = prompt
            self.last_motion_path = str(motion_path)
            self.last_waypoints = [
                {
                    "frame": waypoint["frame"],
                    "position": waypoint["position"],
                    **({"heading": waypoint["heading"]} if waypoint.get("heading") is not None else {}),
                }
                for waypoint in waypoints
            ]

            return {
                "status": "ok",
                "prompt": prompt,
                "duration": duration,
                "fps": fps,
                "requested_new_frames": requested_new_frames,
                "visible_frames": int(visible_motion.shape[1]),
                "history_frames": int(self.motion_tensor.shape[1]) if self.motion_tensor is not None else 0,
                "history_used": bool(previous_end_motion is not None),
                "continuity_root_delta": continuity_delta,
                "waypoints": self.last_waypoints,
                "waypoint_errors": waypoint_errors,
                "motion_path": str(motion_path),
                "blender": blender,
                "render_mp4": render_response,
                "elapsed_seconds": round(time.time() - start, 3),
            }

    def _history_tail(self, model, history_frames: int) -> torch.Tensor | None:
        if self.motion_tensor is None or self.motion_tensor.shape[1] == 0:
            return None
        patch = int(model.num_frames_per_token)
        usable = min(int(self.motion_tensor.shape[1]), int(history_frames))
        usable = (usable // patch) * patch
        if usable < patch:
            return None
        return self.motion_tensor[:, -usable:].detach().clone()

    def _generate_windows(
        self,
        model,
        prompt: str,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        requested_new_frames: int,
        diffusion_steps: int,
        cfg_weight: float | tuple[float, float],
        history_frames: int,
        continue_from_history: bool,
        waypoints: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
        remaining = requested_new_frames
        gen_horizon_len = int(model.gen_horizon_len)
        num_frames_per_token = int(model.num_frames_per_token)
        init_translation = blender_position_to_ardy_translation(self.avatar_position_blender).to(self.config.device)
        init_heading = torch.tensor([self.init_heading], dtype=torch.float32, device=self.config.device)
        new_windows: list[torch.Tensor] = []
        generated_so_far = 0

        while remaining > 0:
            history_tail = self._history_tail(model, history_frames) if continue_from_history else None
            if history_tail is not None:
                history_len = int(history_tail.shape[1])
                total_frames = history_len + gen_horizon_len
                motion_mask, observed_motion = self._build_waypoint_conditions_for_window(
                    model=model,
                    waypoints=waypoints,
                    segment_start_frame=generated_so_far,
                    window_new_frames=gen_horizon_len,
                    history_len=history_len,
                    total_frames=total_frames,
                )
                generated = model.autoregressive_step(
                    num_frames=total_frames,
                    num_denoising_steps=diffusion_steps,
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                    cfg_weight=cfg_weight,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                    init_history_sequence=history_tail,
                    init_global_translation=None,
                    init_first_heading_angle=None,
                )
                new_window = generated[:, history_len:]
            else:
                total_frames = math.ceil(gen_horizon_len / num_frames_per_token) * num_frames_per_token
                motion_mask, observed_motion = self._build_waypoint_conditions_for_window(
                    model=model,
                    waypoints=waypoints,
                    segment_start_frame=generated_so_far,
                    window_new_frames=gen_horizon_len,
                    history_len=0,
                    total_frames=total_frames,
                )
                generated = model.autoregressive_step(
                    num_frames=total_frames,
                    num_denoising_steps=diffusion_steps,
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                    cfg_weight=cfg_weight,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                    init_history_sequence=None,
                    init_global_translation=init_translation,
                    init_first_heading_angle=init_heading,
                )
                new_window = generated[:, :gen_horizon_len]

            take = min(remaining, int(new_window.shape[1]))
            taken = new_window[:, :take].detach()
            if self.motion_tensor is None or not continue_from_history:
                self.motion_tensor = taken.clone()
                continue_from_history = True
            else:
                self.motion_tensor = torch.cat([self.motion_tensor, taken], dim=1)
                if self.motion_tensor.shape[1] > self.max_stored_frames:
                    self.motion_tensor = self.motion_tensor[:, -self.max_stored_frames :].detach()

            new_windows.append(taken)
            remaining -= take
            generated_so_far += take

        if not new_windows:
            raise RuntimeError(f"No motion frames were generated for prompt: {prompt}")
        return new_windows

    def _parse_waypoints(self, raw_waypoints: Any, requested_new_frames: int, fps: float) -> list[dict[str, Any]]:
        if raw_waypoints is None:
            return []
        if not isinstance(raw_waypoints, list):
            raise ValueError("waypoints must be a JSON list.")

        parsed: list[dict[str, Any]] = []
        auto_frame_items: list[tuple[int, Any]] = []
        for index, item in enumerate(raw_waypoints):
            frame: int | None = None
            heading: float | None = None
            position: Any

            if isinstance(item, dict):
                position = item.get("position")
                if position is None:
                    if "x" in item and "y" in item:
                        position = [item["x"], item["y"]]
                    else:
                        raise ValueError(f"Waypoint {index} must include position or x/y.")
                if item.get("frame") is not None:
                    frame = int(item["frame"])
                elif item.get("time") is not None:
                    frame = int(round(float(item["time"]) * fps))
                if item.get("heading") is not None:
                    heading = float(item["heading"])
            elif isinstance(item, list):
                if len(item) == 2:
                    position = item
                elif len(item) == 3:
                    frame = int(item[0])
                    position = item[1:]
                elif len(item) == 4:
                    frame = int(item[0])
                    position = item[1:3]
                    heading = float(item[3])
                else:
                    raise ValueError(
                        "List waypoints must be [x, y], [frame, x, y], or [frame, x, y, heading]."
                    )
            else:
                raise ValueError(f"Unsupported waypoint {index}: expected object or list.")

            parsed_item = {
                "frame": frame,
                "position": blender_ground_to_ardy_root_2d(position),
                "heading": heading,
            }
            if frame is None:
                auto_frame_items.append((len(parsed), parsed_item))
            parsed.append(parsed_item)

        if auto_frame_items:
            count = len(auto_frame_items)
            for auto_idx, (parsed_idx, item) in enumerate(auto_frame_items, start=1):
                frame = int(round(auto_idx * requested_new_frames / count))
                item["frame"] = max(0, min(requested_new_frames - 1, frame))
                parsed[parsed_idx] = item

        normalized = []
        for waypoint in parsed:
            frame = int(waypoint["frame"])
            if not 0 <= frame < requested_new_frames:
                raise ValueError(
                    f"Waypoint frame {frame} is outside generated range [0, {requested_new_frames - 1}]."
                )
            normalized.append(
                {
                    "frame": frame,
                    "position": [float(waypoint["position"][0]), float(waypoint["position"][1])],
                    "heading": waypoint.get("heading"),
                }
            )

        normalized.sort(key=lambda waypoint: waypoint["frame"])
        return normalized

    def _build_waypoint_conditions_for_window(
        self,
        model,
        waypoints: list[dict[str, Any]],
        segment_start_frame: int,
        window_new_frames: int,
        history_len: int,
        total_frames: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        window_waypoints = [
            waypoint
            for waypoint in waypoints
            if segment_start_frame <= int(waypoint["frame"]) < segment_start_frame + window_new_frames
        ]
        if not window_waypoints:
            return None, None

        frame_indices = torch.tensor(
            [history_len + int(waypoint["frame"]) - segment_start_frame for waypoint in window_waypoints],
            device=self.config.device,
            dtype=torch.long,
        )
        root_2d = torch.tensor(
            [waypoint["position"] for waypoint in window_waypoints],
            device=self.config.device,
            dtype=torch.float32,
        )
        headings = (
            torch.tensor(
                [float(waypoint["heading"]) for waypoint in window_waypoints],
                device=self.config.device,
                dtype=torch.float32,
            )
            if all(waypoint.get("heading") is not None for waypoint in window_waypoints)
            else None
        )
        constraints = [Root2DConstraintSet(model.skeleton, frame_indices, root_2d, global_root_heading=headings)]
        observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints(
            constraints,
            length=total_frames,
            to_normalize=True,
            device=self.config.device,
        )
        return motion_mask.unsqueeze(0), observed_motion.unsqueeze(0)

    def _measure_waypoint_errors(
        self,
        visible_output: dict[str, torch.Tensor],
        waypoints: list[dict[str, Any]],
        visible_frame_offset: int,
    ) -> list[dict[str, Any]]:
        if not waypoints:
            return []

        root_positions = visible_output["root_positions"][0]
        errors = []
        for waypoint in waypoints:
            visible_frame = min(int(waypoint["frame"]) + visible_frame_offset, int(root_positions.shape[0]) - 1)
            actual_2d = root_positions[visible_frame, [0, 2]]
            target_2d = torch.tensor(waypoint["position"], device=actual_2d.device, dtype=actual_2d.dtype)
            error = torch.linalg.norm(actual_2d - target_2d).item()
            errors.append(
                {
                    "frame": int(waypoint["frame"]),
                    "visible_frame": visible_frame,
                    "target": waypoint["position"],
                    "actual": ardy_root_2d_to_blender_ground(actual_2d),
                    "error": float(error),
                }
            )
        return errors


class LiveMotionHandler(BaseHTTPRequestHandler):
    server_version = "ARDYLiveMotionAPI/0.1"

    def do_OPTIONS(self) -> None:
        self._send_json({"status": "ok"})

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in {"/", "/health"}:
            self._send_json(self.server.session.health())
            return
        if route == "/diagnostics":
            self._send_json(self.server.session.diagnostics())
            return
        self._send_json({"status": "error", "error": f"Unknown route {route}"}, status=404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
            if route == "/scene/load_usd":
                response = self.server.session.load_usd(payload)
            elif route == "/avatar/place":
                response = self.server.session.place_avatar(payload)
            elif route == "/prompt":
                response = self.server.session.prompt(payload)
            elif route == "/reset":
                response = self.server.session.reset(payload)
            elif route == "/render/mp4":
                response = self.server.session._post_blender("/render/mp4", payload, timeout=1200.0)
            else:
                self._send_json({"status": "error", "error": f"Unknown route {route}"}, status=404)
                return
            self._send_json(response)
        except Exception as error:
            if isinstance(error, SetupError):
                print(f"Setup error: {error}", flush=True)
            else:
                traceback.print_exc()
            self._send_json(
                {
                    "status": "error",
                    "error": humanize_setup_error(error),
                    "detail": str(error),
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


class LiveMotionServer(ThreadingHTTPServer):
    session: LiveMotionSession


def main() -> None:
    args = parse_args()
    checkpoints_dir = args.checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
    config = LiveAPIConfig(
        host=args.host,
        port=args.port,
        default_model=args.model,
        device=args.device or default_device(),
        checkpoints_dir=checkpoints_dir,
        output_dir=args.output_dir.expanduser().resolve(),
        blender_url=args.blender_url.rstrip("/"),
        render_width=args.render_width,
        render_height=args.render_height,
        render_mode=args.render_mode,
        text_encoder_mode=args.text_encoder_mode,
        text_encoder_url=args.text_encoder_url,
        text_encoder_fp32=args.text_encoder_fp32,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    session = LiveMotionSession(config)
    if not args.lazy_load:
        session.preload()

    server = LiveMotionServer((config.host, config.port), LiveMotionHandler)
    server.session = session
    print(f"ARDY live motion API running at http://{config.host}:{config.port}", flush=True)
    print(f"Driving persistent Blender server at {config.blender_url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
