#!/usr/bin/env python3
"""Long-running HTTP API for ARDY text-to-motion generation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

from ardy.constraints import load_constraints_lst
from ardy.model import DEFAULT_MODEL
from ardy.model.load_model import load_model, load_text_encoder
from ardy.model.loading import get_env_var
from ardy.model.registry import resolve_model_name
from ardy.motion_rep.tools import length_to_mask
from ardy.postprocess import post_process_motion
from ardy.skeleton import G1Skeleton34, SOMASkeleton30
from ardy.tools import seed_everything, to_numpy


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_BLENDER = "~/Projects/blender/blender"
LLAMA_BASE_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"


@dataclass
class APIConfig:
    host: str
    port: int
    default_model: str
    device: str
    checkpoints_dir: str | None
    output_dir: Path
    blender: Path | None
    render_by_default: bool
    render_width: int
    render_height: int
    render_mode: str
    text_encoder_mode: str | None
    text_encoder_url: str | None
    text_encoder_fp32: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ARDY prompt-to-motion HTTP API.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind. Default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind. Default: {DEFAULT_PORT}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Default model nickname or full model name.")
    parser.add_argument(
        "--device",
        default=None,
        help='Device, e.g. "cuda:0" or "cpu". Default: cuda:0 if available.',
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        help="Local checkpoint directory. Defaults to CHECKPOINTS_DIR.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/api"), help="Directory for generated files.")
    parser.add_argument(
        "--blender",
        type=Path,
        default=None,
        help=f"Blender executable. Default: {DEFAULT_BLENDER} if present.",
    )
    parser.add_argument(
        "--render-by-default",
        action="store_true",
        help="Render every generation unless request has render=false.",
    )
    parser.add_argument("--render-width", type=int, default=1280, help="Default Blender render width.")
    parser.add_argument("--render-height", type=int, default=720, help="Default Blender render height.")
    parser.add_argument(
        "--render-mode",
        choices=("auto", "skin", "skeleton", "both"),
        default="auto",
        help="Default Blender render mode. Default: auto.",
    )
    parser.add_argument("--text-encoder-mode", choices=("auto", "api", "local"), default=None)
    parser.add_argument("--text-encoder-url", default=None)
    parser.add_argument("--text-encoder-fp32", action="store_true")
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Start server without loading the default model first.",
    )
    return parser.parse_args()


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_blender(path: Path | None) -> Path | None:
    candidates = []
    if path is not None:
        candidates.append(path.expanduser())
    candidates.append(Path(DEFAULT_BLENDER).expanduser())
    found_on_path = shutil.which("blender")
    if found_on_path:
        candidates.append(Path(found_on_path))
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def default_output_name(prompt: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", prompt.lower())[:6]
    slug = "_".join(words) or "motion"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{slug}"


def resolve_output_base(output: str | None, prompt: str, output_dir: Path) -> Path:
    output = output or default_output_name(prompt)
    path = Path(output).expanduser()
    if not path.is_absolute():
        if path.parent == Path("."):
            path = output_dir / path
        else:
            path = Path.cwd() / path
    return path


def ensure_single_file_path(path: Path, suffix: str) -> Path:
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sample_file_paths(base: Path, num_samples: int, suffix: str) -> list[Path]:
    if num_samples == 1:
        return [ensure_single_file_path(base, suffix)]

    folder = base.with_suffix("") if base.suffix else base
    folder.mkdir(parents=True, exist_ok=True)
    base_name = folder.name
    return [folder / f"{base_name}_{idx:02d}{suffix}" for idx in range(num_samples)]


def select_sample(output: dict[str, Any], index: int, num_samples: int) -> dict[str, Any]:
    return {
        key: (
            value[index]
            if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == num_samples
            else value
        )
        for key, value in output.items()
    }


def save_motion_npz(path: Path, motion_dict: dict[str, Any], fps: float, text: str) -> None:
    arrays = {key: np.asarray(value) for key, value in motion_dict.items()}
    arrays["fps"] = np.asarray(fps)
    arrays["text"] = np.asarray(text)
    np.savez(path, **arrays)


def default_history_frames(fps: float, gen_horizon_len: int, num_frames_per_token: int) -> int:
    max_window_len = (int(10 * fps) // num_frames_per_token) * num_frames_per_token
    return ((max_window_len - gen_horizon_len) // num_frames_per_token) * num_frames_per_token


def parse_cfg_weight(value: Any) -> float | tuple[float, float]:
    if value is None:
        return (2.0, 2.0)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 1:
        return float(value[0])
    if isinstance(value, list) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError("cfg_weight must be a number, a one-item list, or [text_weight, constraint_weight].")


class SetupError(RuntimeError):
    """Expected environment/setup problem that should be reported without a dependency traceback."""


def _exception_chain_text(error: BaseException) -> str:
    parts = [f"{type(error).__name__}: {error}"]
    seen = {id(error)}
    current = error
    while current.__cause__ is not None or current.__context__ is not None:
        current = current.__cause__ or current.__context__
        if id(current) in seen:
            break
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
    return "\n".join(parts)


def humanize_setup_error(error: BaseException) -> str:
    text = _exception_chain_text(error)
    lower = text.lower()
    if LLAMA_BASE_REPO.lower() in lower and (
        "403" in lower or "forbidden" in lower or "gated" in lower or "access" in lower
    ):
        return (
            f"Cannot load the ARDY text encoder because Hugging Face denied access to {LLAMA_BASE_REPO}. "
            "Use a Hugging Face token that has public gated repository access enabled and has accepted the "
            "Meta Llama 3 8B Instruct model terms, or run a compatible text-encoder service and start this "
            "API with --text-encoder-mode api --text-encoder-url <url>."
        )
    return str(error)


def command_available(path_or_command: str | Path | None) -> dict[str, Any]:
    if path_or_command is None:
        return {"ok": False, "path": None}
    path = str(path_or_command)
    resolved = shutil.which(path) if os.path.sep not in path else path
    ok = bool(resolved and Path(resolved).exists() and os.access(resolved, os.X_OK))
    return {"ok": ok, "path": str(Path(resolved).resolve()) if resolved and Path(resolved).exists() else resolved}


class MotionGenerator:
    def __init__(self, config: APIConfig):
        self.config = config
        self._lock = threading.Lock()
        self._text_encoder = None
        self._models: dict[str, Any] = {}

    def preload(self) -> None:
        self._load_model(self.config.default_model)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "device": self.config.device,
            "loaded_models": sorted(self._models),
            "default_model": self.config.default_model,
            "blender": str(self.config.blender) if self.config.blender else None,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "python": self._python_diagnostics(),
            "torch": self._torch_diagnostics(),
            "tools": self._tool_diagnostics(),
            "huggingface": self._huggingface_diagnostics(),
            "models": self._model_diagnostics(),
        }

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Request JSON must include a non-empty 'prompt'.")

        with self._lock:
            start = time.time()
            model_name = str(payload.get("model") or self.config.default_model)
            model = self._load_model(model_name)

            if payload.get("seed") is not None:
                seed_everything(int(payload["seed"]))

            duration = float(payload.get("duration", 5.0))
            if duration <= 0:
                raise ValueError("duration must be positive.")

            num_samples = int(payload.get("num_samples", 1))
            if num_samples < 1:
                raise ValueError("num_samples must be >= 1.")

            fps = float(model.motion_rep.fps)
            num_frames = max(1, int(duration * fps))
            diffusion_steps = self._resolve_diffusion_steps(model, payload.get("diffusion_steps"))
            history_frames = self._resolve_history_frames(model, payload.get("history_frames"))
            cfg_weight = parse_cfg_weight(payload.get("cfg_weight"))
            constraints_path = payload.get("constraints")
            postprocess = bool(payload.get("postprocess", True))

            output = self._run_generation(
                model=model,
                prompt=prompt,
                num_frames=num_frames,
                num_samples=num_samples,
                diffusion_steps=diffusion_steps,
                history_frames=history_frames,
                cfg_weight=cfg_weight,
                constraints_path=constraints_path,
                postprocess=postprocess,
            )

            output_base = resolve_output_base(payload.get("output"), prompt, self.config.output_dir).resolve()
            motion_paths, csv_paths = self._save_outputs(model, output, output_base, fps, prompt)

            render = bool(payload.get("render", self.config.render_by_default))
            render_paths: list[str | None] = [None] * len(motion_paths)
            if render:
                render_paths = self._render_outputs(
                    motion_paths=motion_paths,
                    width=int(payload.get("render_width", self.config.render_width)),
                    height=int(payload.get("render_height", self.config.render_height)),
                    render_mode=str(payload.get("render_mode", self.config.render_mode)),
                )

            outputs = []
            for idx, motion_path in enumerate(motion_paths):
                outputs.append(
                    {
                        "motion_path": str(motion_path),
                        "csv_path": str(csv_paths[idx]) if csv_paths[idx] else None,
                        "render_path": render_paths[idx],
                    }
                )

            return {
                "status": "ok",
                "prompt": prompt,
                "model": resolve_model_name(model_name, checkpoints_dir=self.config.checkpoints_dir),
                "duration": duration,
                "fps": fps,
                "num_frames": num_frames,
                "num_samples": num_samples,
                "diffusion_steps": diffusion_steps,
                "history_frames": history_frames,
                "outputs": outputs,
                "elapsed_seconds": round(time.time() - start, 3),
            }

    def _load_model(self, model_name: str):
        resolved = resolve_model_name(model_name, checkpoints_dir=self.config.checkpoints_dir)
        if resolved in self._models:
            return self._models[resolved]

        if self._text_encoder is None:
            try:
                self._text_encoder = load_text_encoder(
                    mode=self.config.text_encoder_mode,
                    url=self.config.text_encoder_url,
                    fp32=self.config.text_encoder_fp32,
                    device=self.config.device,
                )
            except Exception as error:
                raise SetupError(humanize_setup_error(error)) from error

        print(f"Loading ARDY model {resolved} on {self.config.device}...", flush=True)
        model = load_model(
            resolved,
            device=self.config.device,
            checkpoints_dir=self.config.checkpoints_dir,
            text_encoder=self._text_encoder,
        )
        self._models[resolved] = model
        print(f"Loaded ARDY model {resolved}", flush=True)
        return model

    def _resolve_diffusion_steps(self, model, requested: Any) -> int:
        num_base_steps = int(model.diffusion.num_base_steps)
        diffusion_steps = num_base_steps if requested is None else int(requested)
        if not 1 <= diffusion_steps <= num_base_steps:
            raise ValueError(f"diffusion_steps must be between 1 and {num_base_steps}; got {diffusion_steps}.")
        return diffusion_steps

    def _resolve_history_frames(self, model, requested: Any) -> int:
        patch = int(model.num_frames_per_token)
        if requested is None:
            return default_history_frames(float(model.motion_rep.fps), int(model.gen_horizon_len), patch)
        history_frames = int(requested)
        if history_frames < patch or history_frames % patch != 0:
            raise ValueError(f"history_frames must be a positive multiple of {patch}; got {history_frames}.")
        return history_frames

    def _run_generation(
        self,
        model,
        prompt: str,
        num_frames: int,
        num_samples: int,
        diffusion_steps: int,
        history_frames: int,
        cfg_weight: float | tuple[float, float],
        constraints_path: str | None,
        postprocess: bool,
    ) -> dict[str, Any]:
        device = self.config.device
        constraint_lst = load_constraints_lst(constraints_path, model.skeleton) if constraints_path else []
        if constraint_lst:
            max_frame_idx = max(int(constraint.frame_indices.max()) for constraint in constraint_lst)
            if max_frame_idx >= num_frames:
                raise ValueError(
                    f"Constraint frame index {max_frame_idx} exceeds generated length "
                    f"({num_frames} frames); increase duration."
                )

        lengths = torch.tensor([num_frames] * num_samples, device=device)
        pad_mask = length_to_mask(lengths)
        first_heading_angle = torch.zeros(num_samples, device=device)

        observed_motion, motion_mask = None, None
        if constraint_lst:
            observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
                constraint_lst,
                lengths,
                to_normalize=True,
                device=device,
            )

        with torch.no_grad():
            motion = model(
                [prompt] * num_samples,
                num_frames,
                num_denoising_steps=diffusion_steps,
                pad_mask=pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                crop_history_length=history_frames,
            )
            output = model.motion_rep.inverse(motion, is_normalized=True)

        use_postprocess = not isinstance(model.skeleton, G1Skeleton34) and postprocess
        if use_postprocess:
            corrected = post_process_motion(
                output["local_rot_mats"],
                output["root_positions"],
                output["foot_contacts"],
                model.skeleton,
                constraint_lst=constraint_lst or None,
            )
            output.update(corrected)

        if isinstance(model.skeleton, SOMASkeleton30):
            output = model.skeleton.output_to_SOMASkeleton77(output)

        return to_numpy(output)

    def _save_outputs(
        self,
        model,
        output: dict[str, Any],
        output_base: Path,
        fps: float,
        prompt: str,
    ) -> tuple[list[Path], list[Path | None]]:
        num_samples = int(output["posed_joints"].shape[0])
        motion_paths = sample_file_paths(output_base, num_samples, ".npz")
        for idx, path in enumerate(motion_paths):
            save_motion_npz(path, select_sample(output, idx, num_samples), fps, prompt)

        csv_paths: list[Path | None] = [None] * num_samples
        if isinstance(model.skeleton, G1Skeleton34):
            from ardy.exports.mujoco import MujocoQposConverter

            converter = MujocoQposConverter(model.skeleton)
            qpos = converter.dict_to_qpos(output, self.config.device)
            csv_paths = sample_file_paths(output_base, num_samples, ".csv")
            for idx, csv_path in enumerate(csv_paths):
                converter.save_csv(qpos[idx], str(csv_path))
            csv_paths = [path.resolve() for path in csv_paths]

        return [path.resolve() for path in motion_paths], csv_paths

    def _render_outputs(
        self,
        motion_paths: list[Path],
        width: int,
        height: int,
        render_mode: str,
    ) -> list[str | None]:
        if self.config.blender is None:
            raise RuntimeError("Rendering requested, but no Blender executable was found.")

        render_script = Path(__file__).with_name("render_motion_blender.py").resolve()
        render_paths: list[str | None] = []
        for motion_path in motion_paths:
            output_path = motion_path.with_suffix(".mp4")
            cmd = [
                str(self.config.blender),
                "--background",
                "--python",
                str(render_script),
                "--",
                str(motion_path),
                "--output",
                str(output_path),
                "--width",
                str(width),
                "--height",
                str(height),
                "--render-mode",
                render_mode,
            ]
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
                tail = "\n".join(result.stdout.splitlines()[-80:])
                raise RuntimeError(f"Blender render failed for {motion_path}:\n{tail}")
            render_paths.append(str(output_path.resolve()))
        return render_paths

    def _python_diagnostics(self) -> dict[str, Any]:
        import sys

        import ardy

        return {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "ardy_path": getattr(ardy, "__file__", None),
        }

    def _torch_diagnostics(self) -> dict[str, Any]:
        cuda_available = torch.cuda.is_available()
        return {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "device": self.config.device,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        }

    def _tool_diagnostics(self) -> dict[str, Any]:
        blender = command_available(self.config.blender)
        ffmpeg = command_available("ffmpeg")
        return {
            "blender": blender,
            "ffmpeg": ffmpeg,
            "rendering_available": bool(blender["ok"] and ffmpeg["ok"]),
        }

    def _huggingface_diagnostics(self) -> dict[str, Any]:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except Exception as error:
            return {"ok": False, "error": f"huggingface_hub import failed: {error}"}

        result: dict[str, Any] = {}
        try:
            whoami = HfApi().whoami()
            result["token_ok"] = True
            result["user"] = whoami.get("name")
        except Exception as error:
            result["token_ok"] = False
            result["user"] = None
            result["token_error"] = str(error)

        try:
            hf_hub_download(repo_id=LLAMA_BASE_REPO, filename="config.json")
            result["llama_access_ok"] = True
            result["llama_repo"] = LLAMA_BASE_REPO
        except Exception as error:
            result["llama_access_ok"] = False
            result["llama_repo"] = LLAMA_BASE_REPO
            result["llama_error"] = humanize_setup_error(error)

        result["ok"] = bool(result.get("token_ok") and result.get("llama_access_ok"))
        return result

    def _model_diagnostics(self) -> dict[str, Any]:
        try:
            resolved = resolve_model_name(self.config.default_model, checkpoints_dir=self.config.checkpoints_dir)
            return {"default_model": self.config.default_model, "resolved_default_model": resolved, "ok": True}
        except Exception as error:
            return {"default_model": self.config.default_model, "ok": False, "error": str(error)}


class MotionAPIHandler(BaseHTTPRequestHandler):
    server_version = "ARDYMotionAPI/0.1"

    def do_OPTIONS(self) -> None:
        self._send_json({"status": "ok"})

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/health":
            self._send_json(self.server.generator.health())
            return
        if route == "/diagnostics":
            self._send_json(self.server.generator.diagnostics())
            return
        if route == "/":
            self._send_json(
                {
                    "status": "ok",
                    "routes": {
                        "GET /health": "Server status",
                        "GET /diagnostics": "Local setup and Hugging Face access checks",
                        "POST /generate": "Generate motion from JSON {prompt, duration?, model?, render?}",
                    },
                }
            )
            return
        self._send_json({"status": "error", "error": f"Unknown route {route}"}, status=404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route != "/generate":
            self._send_json({"status": "error", "error": f"Unknown route {route}"}, status=404)
            return

        try:
            payload = self._read_json()
            response = self.server.generator.generate(payload)
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
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

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


class MotionAPIServer(ThreadingHTTPServer):
    generator: MotionGenerator


def main() -> None:
    args = parse_args()
    checkpoints_dir = args.checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
    config = APIConfig(
        host=args.host,
        port=args.port,
        default_model=args.model,
        device=args.device or default_device(),
        checkpoints_dir=checkpoints_dir,
        output_dir=args.output_dir.expanduser().resolve(),
        blender=resolve_blender(args.blender),
        render_by_default=args.render_by_default,
        render_width=args.render_width,
        render_height=args.render_height,
        render_mode=args.render_mode,
        text_encoder_mode=args.text_encoder_mode,
        text_encoder_url=args.text_encoder_url,
        text_encoder_fp32=args.text_encoder_fp32,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    generator = MotionGenerator(config)
    if not args.lazy_load:
        generator.preload()

    server = MotionAPIServer((config.host, config.port), MotionAPIHandler)
    server.generator = generator

    print(f"ARDY motion API running at http://{config.host}:{config.port}", flush=True)
    print("POST JSON to /generate with at least {'prompt': '...'}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
