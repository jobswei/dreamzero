#!/usr/bin/env python3
"""Quick local Wan2.2-TI2V-5B DreamZero-style inference demo.

This script is intentionally checkpoint-light:

- it does not require a DreamZero Wan2.2 policy checkpoint
- it loads the Wan2.2-TI2V-5B base DiT/T5/VAE weights directly
- it loads the Wan2.1 CLIP image encoder required by Wan2.2
- DreamZero-specific action/state modules are left randomly initialized

The goal is only to verify that the 5B stack runs end to end on local
debug videos and produces:

- decoded predicted video latents as mp4
- raw action tensor outputs as numpy files
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("DISABLE_TORCH_COMPILE", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHead,
)

VIDEO_FILES = {
    "left": "debug_image/exterior_image_1_left.mp4",
    "right": "debug_image/exterior_image_2_left.mp4",
    "wrist": "debug_image/wrist_image_left.mp4",
}

RELATIVE_OFFSETS = [-23, -16, -8, 0]
NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, "
    "painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, "
    "mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused "
    "fingers, stagnant image, cluttered background, three legs, many people in the background, "
    "walking backwards."
)

DEFAULT_WAN22_CKPT_DIR = (
    "/ssd/firefly/VAM2/work_dirs/Checkpoints/Wan2.2-TI2V-5B"
)
DEFAULT_IMAGE_ENCODER_DIR = (
    "/ssd/firefly/VAM2/work_dirs/Checkpoints/Wan2.1-I2V-14B-480P"
)
DEFAULT_OUTPUT_DIR = "outputs/demo_droid_videos_wan22_base"

NUM_FRAMES = 33
NUM_FRAME_PER_BLOCK = 2
ACTION_HORIZON = 24
ACTION_DIM = 8
MAX_STATE_DIM = 64
NUM_ACTION_PER_BLOCK = 24
NUM_STATE_PER_BLOCK = 1
TARGET_SINGLE_VIEW_HEIGHT = 160
TARGET_SINGLE_VIEW_WIDTH = 320
TARGET_VIDEO_HEIGHT = 160
TARGET_VIDEO_WIDTH = 320
MAX_CHUNK_SIZE = 4
TOKENIZER_MAX_LENGTH = 512


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


def load_image_as_frames(image_path: Path, num_frames: int) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.uint8)
    return np.repeat(image_rgb[None], num_frames, axis=0)


def load_camera_frames(video_root: Path) -> dict[str, np.ndarray]:
    return {
        key: load_all_frames(str(video_root / Path(path).name))
        for key, path in VIDEO_FILES.items()
    }


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def build_composite_frames(
    left_frames: np.ndarray,
    right_frames: np.ndarray,
    wrist_frames: np.ndarray,
) -> np.ndarray:
    if not (len(left_frames) == len(right_frames) == len(wrist_frames)):
        raise ValueError("All camera streams must have the same number of frames")

    composites = []
    for left, right, wrist in zip(left_frames, right_frames, wrist_frames, strict=True):
        left = resize_frame(left, TARGET_SINGLE_VIEW_HEIGHT, TARGET_SINGLE_VIEW_WIDTH)
        right = resize_frame(right, TARGET_SINGLE_VIEW_HEIGHT, TARGET_SINGLE_VIEW_WIDTH)
        wrist = resize_frame(wrist, TARGET_SINGLE_VIEW_HEIGHT, TARGET_SINGLE_VIEW_WIDTH)

        canvas = np.zeros(
            (TARGET_SINGLE_VIEW_HEIGHT * 2, TARGET_SINGLE_VIEW_WIDTH * 2, 3),
            dtype=np.uint8,
        )
        wrist_wide = np.repeat(wrist, 2, axis=1)
        canvas[:TARGET_SINGLE_VIEW_HEIGHT, :] = wrist_wide
        canvas[TARGET_SINGLE_VIEW_HEIGHT:, :TARGET_SINGLE_VIEW_WIDTH] = left
        canvas[TARGET_SINGLE_VIEW_HEIGHT:, TARGET_SINGLE_VIEW_WIDTH:] = right
        composites.append(canvas)
    return np.stack(composites, axis=0)


def build_single_view_frames(frames: np.ndarray) -> np.ndarray:
    resized = [
        resize_frame(frame, TARGET_VIDEO_HEIGHT, TARGET_VIDEO_WIDTH)
        for frame in frames
    ]
    return np.stack(resized, axis=0).astype(np.uint8)


def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current_frame = ACTION_HORIZON - 1
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def format_droid_prompt(prompt: str) -> str:
    prompt = prompt.strip().lower()
    return (
        "A multi-view video shows that a robot "
        + prompt
        + " The video is split into three views: The top view shows the camera view from the "
        + "robot's wrist, the bottom-left view shows the camera view from the left exterior "
        + "camera, and the bottom-right view shows the camera view from the right exterior camera. "
        + "During training, one of the two bottom exterior views may be a black screen "
        + "(dropped view). The robot "
        + prompt
    )


def tokenize_text(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = TOKENIZER_MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return encoded.input_ids, encoded.attention_mask


def load_action_head_config(
    wan22_ckpt_dir: Path,
    image_encoder_dir: Path,
) -> dict:
    cfg = OmegaConf.load("groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml")
    wan22_cfg = OmegaConf.load(
        "groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22.yaml"
    )
    cfg = OmegaConf.merge(cfg, wan22_cfg)

    cfg.num_frames = NUM_FRAMES
    cfg.num_frame_per_block = NUM_FRAME_PER_BLOCK
    cfg.max_state_dim = MAX_STATE_DIM
    cfg.max_action_dim = ACTION_DIM
    cfg.action_horizon = ACTION_HORIZON
    cfg.backbone_hidden_size = 0
    cfg.max_chunk_size = MAX_CHUNK_SIZE
    cfg.num_action_per_block = NUM_ACTION_PER_BLOCK
    cfg.num_state_per_block = NUM_STATE_PER_BLOCK
    cfg.train_architecture = "full"
    cfg.dit_version = str(wan22_ckpt_dir)
    cfg.text_encoder_pretrained_path = str(
        wan22_ckpt_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    cfg.image_encoder_pretrained_path = str(
        image_encoder_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    cfg.vae_pretrained_path = str(wan22_ckpt_dir / "Wan2.2_VAE.pth")

    cfg.action_head_cfg.config.train_architecture = "full"
    cfg.action_head_cfg.config.use_gradient_checkpointing = False
    cfg.action_head_cfg.config.tune_projector = True
    cfg.action_head_cfg.config.tune_diffusion_model = False
    cfg.action_head_cfg.config.num_inference_timesteps = 4
    cfg.action_head_cfg.config.action_dim = ACTION_DIM
    cfg.action_head_cfg.config.diffusion_model_cfg.action_dim = ACTION_DIM

    resolved = OmegaConf.to_container(cfg.action_head_cfg, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def materialize_random_module(module: torch.nn.Module) -> None:
    for child in module.modules():
        for name, param in list(child._parameters.items()):
            if param is None or not getattr(param, "is_meta", False):
                continue
            tensor = torch.empty(param.shape, dtype=param.dtype, device="cpu")
            if name.endswith("bias"):
                torch.nn.init.zeros_(tensor)
            elif "norm" in child.__class__.__name__.lower() or name in {"weight", "gamma"} and tensor.ndim == 1:
                torch.nn.init.ones_(tensor)
            elif tensor.ndim >= 2:
                torch.nn.init.xavier_uniform_(tensor)
            else:
                torch.nn.init.ones_(tensor)
            child._parameters[name] = torch.nn.Parameter(
                tensor,
                requires_grad=param.requires_grad,
            )

        for name, buf in list(child._buffers.items()):
            if buf is None or not getattr(buf, "is_meta", False):
                continue
            child._buffers[name] = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")


def count_meta_tensors(module: torch.nn.Module) -> int:
    total = 0
    for _, param in module.named_parameters():
        if getattr(param, "is_meta", False):
            total += 1
    for _, buf in module.named_buffers():
        if getattr(buf, "is_meta", False):
            total += 1
    return total


def initialize_missing_action_modules(action_head: WANPolicyHead) -> None:
    materialize_random_module(action_head.model)


def build_action_input(
    composite_frames: np.ndarray,
    frame_indices: list[int],
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
) -> BatchFeature:
    selected = composite_frames[frame_indices]
    images = torch.from_numpy(selected[None]).to(device=device)
    state = torch.zeros((1, 1, MAX_STATE_DIM), dtype=torch.bfloat16, device=device)
    embodiment_id = torch.zeros((1,), dtype=torch.long, device=device)

    text_prompt = format_droid_prompt(prompt)
    text_ids, text_mask = tokenize_text(tokenizer, [text_prompt])
    neg_ids, neg_mask = tokenize_text(tokenizer, [NEGATIVE_PROMPT])

    return BatchFeature(
        data={
            "images": images,
            "state": state,
            "embodiment_id": embodiment_id,
            "text": text_ids.to(device=device),
            "text_attention_mask": text_mask.to(device=device),
            "text_negative": neg_ids.to(device=device),
            "text_attention_mask_negative": neg_mask.to(device=device),
        }
    )


def decode_video_prediction(action_head: WANPolicyHead, video_pred: torch.Tensor) -> np.ndarray:
    action_head._ensure_vae_on_device(video_pred)
    with torch.inference_mode():
        decoded = action_head.vae.decode(
            video_pred.to(dtype=torch.bfloat16),
            tiled=action_head.tiled,
            tile_size=(action_head.tile_size_height, action_head.tile_size_width),
            tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
        )
    decoded = decoded.detach().float().cpu().clamp_(-1, 1)
    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    return decoded.permute(0, 2, 3, 4, 1).numpy()


def append_video_segment(segments: list[np.ndarray], segment: np.ndarray) -> None:
    if not segments:
        segments.append(segment)
        return
    prev = segments[-1]
    if prev.shape[1:] == segment.shape[1:] and np.array_equal(prev[-1], segment[0]):
        segment = segment[1:]
    if len(segment) > 0:
        segments.append(segment)


def normalize_action_array(value: torch.Tensor) -> np.ndarray:
    arr = value.detach().float().cpu().numpy()
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected action array with shape [T, D], got {arr.shape}")
    return arr


def save_action_outputs(action: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "combined_actions.npy", action)
    np.savez(output_dir / "combined_actions_by_key.npz", action_pred=action)
    with open(output_dir / "combined_actions_meta.json", "w") as f:
        json.dump({"action_pred": [0, int(action.shape[1])]}, f, indent=2)


def save_video_mp4(video: np.ndarray, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output_path, video, fps=fps)


def log_action(step_name: str, action_pred: torch.Tensor, dt: float) -> None:
    arr = normalize_action_array(action_pred)
    print(
        f"{step_name}: {dt:.2f}s, action_pred shape={arr.shape}, "
        f"range=[{arr.min():.4f}, {arr.max():.4f}]"
    )


def assert_required_files(wan22_ckpt_dir: Path, image_encoder_dir: Path, tokenizer_path: Path) -> None:
    required = [
        wan22_ckpt_dir / "diffusion_pytorch_model.safetensors.index.json",
        wan22_ckpt_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        wan22_ckpt_dir / "Wan2.2_VAE.pth",
        image_encoder_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        tokenizer_path / "tokenizer.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=DEFAULT_WAN22_CKPT_DIR,
        help="Alias for Wan2.2-TI2V-5B base checkpoint directory.",
    )
    parser.add_argument(
        "--wan22-ckpt-dir",
        default=None,
        help="Optional explicit Wan2.2-TI2V-5B base checkpoint directory.",
    )
    parser.add_argument(
        "--image-encoder-dir",
        default=DEFAULT_IMAGE_ENCODER_DIR,
        help="Directory containing Wan2.1 CLIP weights used by Wan2.2.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer path override. Defaults to <wan22-ckpt-dir>/google/umt5-xxl.",
    )
    parser.add_argument(
        "--video-root",
        default="debug_image",
        help="Directory containing local debug mp4 files.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Optional single image path. When set, the image is repeated as a single-view sequence.",
    )
    parser.add_argument(
        "--device",
        default="cuda:6" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Move the pan forward and use the brush in the middle of the plates "
            "to brush the inside of the pan"
        ),
        help="Instruction text.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="How many 4-frame chunks to run after the initial 1-frame warmup call.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="FPS used when saving predicted videos.",
    )
    parser.add_argument(
        "--save-video-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save decoded predicted videos and raw actions.",
    )
    args = parser.parse_args()

    wan22_ckpt_dir = Path(args.wan22_ckpt_dir or args.model_path).resolve()
    image_encoder_dir = Path(args.image_encoder_dir).resolve()
    tokenizer_path = Path(args.tokenizer_path).resolve() if args.tokenizer_path else wan22_ckpt_dir / "google" / "umt5-xxl"
    video_root = Path(args.video_root).resolve()
    image_path = Path(args.image_path).resolve() if args.image_path else None

    assert_required_files(wan22_ckpt_dir, image_encoder_dir, tokenizer_path)
    if image_path is None:
        for rel_path in VIDEO_FILES.values():
            expected = video_root / Path(rel_path).name
            if not expected.exists():
                raise FileNotFoundError(f"Missing input video: {expected}")
    elif not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"Using device: {device}")
    print(f"Using Wan2.2 base checkpoint: {wan22_ckpt_dir}")
    print(f"Using CLIP image encoder dir: {image_encoder_dir}")
    print(f"Using tokenizer path: {tokenizer_path}")
    if image_path is None:
        print(f"Reading debug videos from: {video_root}")
        camera_frames = load_camera_frames(video_root)
        input_frames = build_composite_frames(
            camera_frames["left"],
            camera_frames["right"],
            camera_frames["wrist"],
        )
    else:
        print(f"Reading single image from: {image_path}")
        repeated = load_image_as_frames(image_path, ACTION_HORIZON)
        input_frames = build_single_view_frames(repeated)
    total_frames = input_frames.shape[0]
    chunks = build_frame_schedule(total_frames, args.num_chunks)
    print(
        f"Loaded input frames: {input_frames.shape}, "
        f"running {1 + len(chunks)} inference calls."
    )

    config = load_action_head_config(wan22_ckpt_dir, image_encoder_dir)
    print("Instantiating Wan2.2 action head.")
    action_head = instantiate(config)
    assert isinstance(action_head, WANPolicyHead)
    initialize_missing_action_modules(action_head)
    action_head.eval()
    action_head.requires_grad_(False)
    action_head.post_initialize()

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
    )

    dummy_backbone = BatchFeature(data={})
    video_segments: list[np.ndarray] = []
    action_segments: list[np.ndarray] = []

    warmup_input = build_action_input(
        composite_frames=input_frames,
        frame_indices=[0],
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
    )

    t0 = time.perf_counter()
    with torch.inference_mode():
        model_pred = action_head.lazy_joint_video_action(dummy_backbone, warmup_input)
    dt = time.perf_counter() - t0
    log_action("initial [0]", model_pred["action_pred"], dt)
    decoded_video = decode_video_prediction(action_head, model_pred["video_pred"])
    append_video_segment(video_segments, decoded_video[0])
    action_segments.append(normalize_action_array(model_pred["action_pred"]))

    for i, frame_indices in enumerate(chunks):
        action_input = build_action_input(
            composite_frames=input_frames,
            frame_indices=frame_indices,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
        )
        t0 = time.perf_counter()
        with torch.inference_mode():
            model_pred = action_head.lazy_joint_video_action(dummy_backbone, action_input)
        dt = time.perf_counter() - t0
        log_action(f"chunk {i} {frame_indices}", model_pred["action_pred"], dt)
        decoded_video = decode_video_prediction(action_head, model_pred["video_pred"])
        append_video_segment(video_segments, decoded_video[0])
        action_segments.append(normalize_action_array(model_pred["action_pred"]))

    combined_video = np.concatenate(video_segments, axis=0)
    combined_actions = np.concatenate(action_segments, axis=0)

    output_dir = Path(args.save_video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_video_mp4(combined_video, output_dir / "combined.mp4", args.fps)
    save_action_outputs(combined_actions, output_dir)

    print(f"Saved combined predicted video: {output_dir / 'combined.mp4'}")
    print(f"Saved raw action tensor: {output_dir / 'combined_actions.npy'}")
    print(f"Target video resize in action head: {TARGET_VIDEO_HEIGHT}x{TARGET_VIDEO_WIDTH}")
    print("Note: action outputs come from randomly initialized DreamZero action/state modules.")


if __name__ == "__main__":
    main()
