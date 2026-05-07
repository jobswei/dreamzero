#!/usr/bin/env python3
"""Minimal local DreamZero-DROID demo inference.

Runs DreamZero-DROID directly without the websocket server/client split.
It reuses the repo's debug mp4 files and the same frame schedule as
test_client_AR.py, then prints the predicted action chunk shapes/timings.
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
from tianshou.data import Batch

torch._dynamo.config.disable = True

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

VIDEO_FILES = {
    "video.exterior_image_1_left": "debug_image/exterior_image_1_left.mp4",
    "video.exterior_image_2_left": "debug_image/exterior_image_2_left.mp4",
    "video.wrist_image_left": "debug_image/wrist_image_left.mp4",
}

RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def init_dist() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(find_free_port()))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, world_size=1, rank=0)


def load_all_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def load_camera_frames() -> dict[str, np.ndarray]:
    return {key: load_all_frames(path) for key, path in VIDEO_FILES.items()}


def build_frame_schedule(total_frames: int,
                         num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current_frame = 23
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def make_obs(camera_frames: dict[str, np.ndarray], frame_indices: list[int],
             prompt: str) -> dict:
    obs: dict = {}
    for key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        if len(frame_indices) == 1:
            selected = selected[0]
        obs[key] = selected

    obs["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
    obs["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
    obs["annotation.language.language_instruction"] = prompt
    obs["annotation.language.language_instruction_2"] = prompt
    obs["annotation.language.language_instruction_3"] = prompt
    return obs


def log_action(step_name: str, result_batch: Batch, dt: float) -> None:
    action_dict = result_batch.act
    print(f"{step_name}: {dt:.2f}s")
    for key in sorted(action_dict.keys()):
        value = action_dict[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        print(f"  {key}: shape={value.shape}, "
              f"range=[{value.min():.4f}, {value.max():.4f}]")


def decode_video_prediction(policy: GrootSimPolicy,
                            video_pred: torch.Tensor) -> np.ndarray:
    action_head = policy.trained_model.action_head
    action_head._ensure_vae_on_device(video_pred)
    with torch.inference_mode():
        decoded = action_head.vae.decode(
            video_pred.to(device=policy.device, dtype=torch.bfloat16),
            tiled=action_head.tiled,
            tile_size=(action_head.tile_size_height, action_head.tile_size_width),
            tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
        )
    decoded = decoded.detach().float().cpu().clamp_(-1, 1)
    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 4, 1).numpy()
    return decoded


def save_video_mp4(video: np.ndarray, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = video
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames with shape [T, H, W, C], got {frames.shape}")
    imageio.mimwrite(output_path, frames, fps=fps)


def append_video_segment(segments: list[np.ndarray], segment: np.ndarray) -> None:
    if segment.ndim != 4:
        raise ValueError(f"Expected video segment shape [T, H, W, C], got {segment.shape}")
    if not segments:
        segments.append(segment)
        return
    prev = segments[-1]
    if prev.shape[1:] == segment.shape[1:] and np.array_equal(prev[-1], segment[0]):
        segment = segment[1:]
    if len(segment) > 0:
        segments.append(segment)


def normalize_action_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected action array with shape [T, D], got {arr.shape}")
    return arr


def extract_action_segment(result_batch: Batch) -> dict[str, np.ndarray]:
    return {
        key: normalize_action_array(value)
        for key, value in result_batch.act.items()
    }


def concat_action_segments(segments: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not segments:
        return {}
    keys = sorted(segments[0].keys())
    return {
        key: np.concatenate([segment[key] for segment in segments], axis=0)
        for key in keys
    }


def flatten_action_dict(action_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, list[int]]]:
    flat_parts = []
    metadata: dict[str, list[int]] = {}
    start = 0
    for key in sorted(action_dict.keys()):
        value = action_dict[key]
        end = start + value.shape[1]
        metadata[key] = [start, end]
        flat_parts.append(value)
        start = end
    if not flat_parts:
        return np.empty((0, 0), dtype=np.float32), metadata
    return np.concatenate(flat_parts, axis=1), metadata


def save_action_outputs(action_dict: dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "combined_actions_by_key.npz", **action_dict)
    flat_action, metadata = flatten_action_dict(action_dict)
    np.save(output_dir / "combined_actions.npy", flat_action)
    with open(output_dir / "combined_actions_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/ssd/firefly/Checkpoints/DreamZero-DROID",
        help="Path to DreamZero-DROID checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="/ssd/firefly/Checkpoints/Wan2.1-I2V-14B-480P/google/umt5-xxl",
        help="Local tokenizer path override.",
    )
    parser.add_argument(
        "--wan-ckpt-dir",
        default="/ssd/firefly/Checkpoints/Wan2.1-I2V-14B-480P",
        help="Local Wan2.1 checkpoint directory used to override action-head pretrained component paths.",
    )
    parser.add_argument(
        "--device",
        default="cuda:6" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--prompt",
        default=
        "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
        help="Instruction text.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=3,
        help="How many 4-frame chunks to run after the initial frame.",
    )
    parser.add_argument(
        "--save-video-dir",
        default="outputs/demo_droid_videos",
        help="Directory to save decoded predicted videos as mp4 files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="FPS used when saving predicted videos.",
    )
    args = parser.parse_args()

    init_dist()
    if args.device.startswith("cuda"):
        device_index = int(
            args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_index)

    print(f"Using device: {args.device}")
    print(f"Loading checkpoint from: {args.model_path}")
    print(f"Using tokenizer path: {args.tokenizer_path}")
    print(f"Using Wan checkpoint dir: {args.wan_ckpt_dir}")
    print(f"Saving predicted videos to: {args.save_video_dir}")
    print(
        "Starting GrootSimPolicy load. This checkpoint is large and may take a while."
    )

    model_config_overrides = [
        f"action_head_cfg.config.diffusion_model_cfg.diffusion_model_pretrained_path={args.wan_ckpt_dir}",
        f"action_head_cfg.config.text_encoder_cfg.text_encoder_pretrained_path={args.wan_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth",
        f"action_head_cfg.config.image_encoder_cfg.image_encoder_pretrained_path={args.wan_ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        f"action_head_cfg.config.vae_cfg.vae_pretrained_path={args.wan_ckpt_dir}/Wan2.1_VAE.pth",
        "action_head_cfg.config.skip_component_loading=true",
    ]

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.OXE_DROID,
        model_path=args.model_path,
        tokenizer_path_override=args.tokenizer_path,
        model_config_overrides=model_config_overrides,
        device=args.device,
    )
    print("Policy loaded.")

    camera_frames = load_camera_frames()
    total_frames = min(v.shape[0] for v in camera_frames.values())
    chunks = build_frame_schedule(total_frames, args.num_chunks)
    print(
        f"Loaded debug videos, total frames={total_frames}, running {1 + len(chunks)} inference calls."
    )

    t0 = time.perf_counter()
    video_segments: list[np.ndarray] = []
    action_segments: list[dict[str, np.ndarray]] = []

    result, video_pred = policy.lazy_joint_forward_causal(
        Batch(obs=make_obs(camera_frames, [0], args.prompt)))
    log_action("initial [0]", result, time.perf_counter() - t0)
    decoded_video = decode_video_prediction(policy, video_pred)
    append_video_segment(video_segments, decoded_video[0])
    action_segments.append(extract_action_segment(result))

    for i, frame_indices in enumerate(chunks):
        t0 = time.perf_counter()
        result, video_pred = policy.lazy_joint_forward_causal(
            Batch(obs=make_obs(camera_frames, frame_indices, args.prompt)))
        log_action(f"chunk {i} {frame_indices}", result,
                   time.perf_counter() - t0)
        decoded_video = decode_video_prediction(policy, video_pred)
        append_video_segment(video_segments, decoded_video[0])
        action_segments.append(extract_action_segment(result))

    output_dir = Path(args.save_video_dir)
    combined_video = np.concatenate(video_segments, axis=0)
    combined_video_path = output_dir / "combined.mp4"
    save_video_mp4(combined_video, combined_video_path, args.fps)
    print(f"Saved combined predicted video: {combined_video_path}")

    combined_actions = concat_action_segments(action_segments)
    save_action_outputs(combined_actions, output_dir)
    print(f"Saved combined actions: {output_dir / 'combined_actions_by_key.npz'}")
    print(f"Saved flattened combined actions: {output_dir / 'combined_actions.npy'}")


if __name__ == "__main__":
    main()
