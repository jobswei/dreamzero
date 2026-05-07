#!/usr/bin/env python3
"""Minimal local DreamZero-AgiBot demo inference.

Loads the DreamZero-AgiBot checkpoint directly, builds one fake AgiBot-style
observation from the repo's debug mp4 files, runs a single forward pass, and
prints the predicted action keys/shapes.
"""

import argparse
import os
import socket

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


VIDEO_FILES = {
    "video.top_head": "debug_image/exterior_image_1_left.mp4",
    "video.hand_left": "debug_image/exterior_image_2_left.mp4",
    "video.hand_right": "debug_image/wrist_image_left.mp4",
}

STATE_DIMS = {
    "state.left_arm_joint_position": 7,
    "state.right_arm_joint_position": 7,
    "state.left_effector_position": 1,
    "state.right_effector_position": 1,
    "state.head_position": 2,
    "state.waist_pitch": 1,
    "state.waist_lift": 1,
}


def init_dist() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(find_free_port()))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, world_size=1, rank=0)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def read_frames(video_path: str, num_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to load frames from {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames, axis=0).astype(np.uint8)


def build_obs(prompt: str, num_frames: int) -> dict:
    obs: dict = {}
    for key, rel_path in VIDEO_FILES.items():
        obs[key] = read_frames(rel_path, num_frames)
    for key, dim in STATE_DIMS.items():
        obs[key] = np.zeros((1, dim), dtype=np.float64)
    obs["annotation.language.action_text"] = prompt
    return obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="./checkpoints/DreamZero-AgiBot",
        help="Path to DreamZero-AgiBot checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="./checkpoints/Wan2.1-I2V-14B-480P/google/umt5-xxl",
        help="Local tokenizer path used to override the absolute path in the checkpoint config.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--prompt",
        default="pick up the object",
        help="Instruction text.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of frames per camera to feed. AgiBot eval expects 4 history frames.",
    )
    args = parser.parse_args()

    init_dist()
    if args.device.startswith("cuda"):
        device_index = int(args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_index)

    print(f"Using device: {args.device}")
    print(f"Loading checkpoint from: {args.model_path}")
    print(f"Using tokenizer path: {args.tokenizer_path}")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.AGIBOT,
        model_path=args.model_path,
        tokenizer_path_override=args.tokenizer_path,
        device=args.device,
    )
    print("Policy loaded.")

    obs = build_obs(prompt=args.prompt, num_frames=args.num_frames)
    print("Observation built. Starting inference...")

    with torch.inference_mode():
        result, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))
    print("Inference finished.")

    print("Predicted action keys:")
    for key in sorted(result.act.keys()):
        value = result.act[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        print(f"  {key}: shape={value.shape}, min={value.min():.4f}, max={value.max():.4f}")


if __name__ == "__main__":
    main()
