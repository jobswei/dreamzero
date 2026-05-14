#!/usr/bin/env python3
"""Local DreamZero-AgiBot demo inference on EWM meta samples.

This follows the same direct-policy path as ``scripts/demo_droid_infer.py``
but adapts the input format to the released DreamZero-AgiBot checkpoint.

Notes:
- The released AgiBot checkpoint's own ``experiment_cfg/conf.yaml`` is treated
  as the source of truth for modality keys and horizons.
- The EWM meta sample directory provides videos and instructions, but not the
  20D DreamZero state vector directly. By default this script uses the AgiBot
  checkpoint metadata mean-state as a stable fallback. If the raw AgiBot
  proprio source is available, ``--raw-agibot-root`` can be used to recover
  per-frame state from ``proprio_stats.h5``.
"""

import argparse
import json
import os
import pickle
import socket
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
from tianshou.data import Batch

torch._dynamo.config.disable = True

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


STATE_SLICES = {
    "state.left_arm_joint_position": slice(0, 7),
    "state.right_arm_joint_position": slice(7, 14),
    "state.left_effector_position": slice(14, 15),
    "state.right_effector_position": slice(15, 16),
    "state.head_position": slice(16, 18),
    "state.waist_pitch": slice(18, 19),
    "state.waist_lift": slice(19, 20),
}

ACTION_KEY_ORDER = [
    "action.left_arm_joint_position",
    "action.right_arm_joint_position",
    "action.left_effector_position",
    "action.right_effector_position",
    "action.head_position",
    "action.waist_pitch",
    "action.waist_lift",
    "action.robot_velocity",
]

VIDEO_KEY_MAP = {
    "head": "video.top_head",
    "hand_left": "video.hand_left",
    "hand_right": "video.hand_right",
}

# The shared WAN causal action head expects a 4-frame observation window for
# continuation calls, matching the DROID AR demo / server path.
CAUSAL_OBS_FRAMES = 4


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


def load_all_frames(video_path: Path) -> np.ndarray:
    # AV1 clips in the EWM meta samples can fail in OpenCV on some machines.
    try:
        import decord

        vr = decord.VideoReader(str(video_path))
        frames = [vr[i].asnumpy() for i in range(len(vr))]
        if frames:
            return np.stack(frames, axis=0).astype(np.uint8)
    except Exception:
        pass

    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        frames = [frame for frame in reader]
        reader.close()
        if frames:
            return np.stack(frames, axis=0).astype(np.uint8)
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
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


def select_episode(annotations_path: Path, episode_index: int) -> dict:
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    if not isinstance(annotations, list) or not annotations:
        raise RuntimeError(f"No episodes found in {annotations_path}")
    if episode_index < 0 or episode_index >= len(annotations):
        raise IndexError(
            f"episode_index={episode_index} out of range for {annotations_path} "
            f"(num_episodes={len(annotations)})"
        )
    return annotations[episode_index]


def load_episode_assets(dataset_root: Path, annotation: dict) -> tuple[dict[str, np.ndarray], str]:
    video_paths = annotation.get("videos", {})
    if not video_paths:
        raise RuntimeError("annotation has no 'videos' field")

    camera_frames: dict[str, np.ndarray] = {}
    for ann_key, rel_path in video_paths.items():
        if ann_key not in VIDEO_KEY_MAP:
            continue
        abs_path = dataset_root / rel_path
        camera_frames[VIDEO_KEY_MAP[ann_key]] = load_all_frames(abs_path)

    missing = sorted(set(VIDEO_KEY_MAP.values()) - set(camera_frames.keys()))
    if missing:
        raise RuntimeError(f"Missing video streams for episode: {missing}")

    instructions = annotation.get("instructions") or []
    prompt = instructions[0] if instructions else annotation.get("meta", {}).get("task_name", "")
    if not prompt:
        prompt = "Execute the task shown in the video."
    return camera_frames, prompt


def load_camera_metadata(dataset_root: Path, annotation: dict) -> dict | None:
    rel_path = annotation.get("camera")
    if not rel_path:
        return None
    camera_path = dataset_root / rel_path
    if not camera_path.exists():
        return None
    with open(camera_path, "rb") as f:
        return pickle.load(f)


def load_checkpoint_metadata(model_path: Path) -> dict:
    metadata_path = model_path / "experiment_cfg" / "metadata.json"
    with open(metadata_path, "r") as f:
        return json.load(f)


def build_mean_state_from_metadata(checkpoint_metadata: dict) -> np.ndarray:
    stats = checkpoint_metadata["agibot"]["statistics"]["state"]
    parts = [
        np.asarray(stats["left_arm_joint_position"]["mean"], dtype=np.float64),
        np.asarray(stats["right_arm_joint_position"]["mean"], dtype=np.float64),
        np.asarray(stats["left_effector_position"]["mean"], dtype=np.float64),
        np.asarray(stats["right_effector_position"]["mean"], dtype=np.float64),
        np.asarray(stats["head_position"]["mean"], dtype=np.float64),
        np.asarray(stats["waist_pitch"]["mean"], dtype=np.float64),
        np.asarray(stats["waist_lift"]["mean"], dtype=np.float64),
    ]
    state = np.concatenate(parts, axis=0)
    if state.shape != (20,):
        raise RuntimeError(f"Unexpected AgiBot mean-state shape: {state.shape}")
    return state


def load_states_from_raw_root(raw_root: Path, annotation: dict) -> np.ndarray | None:
    origin_path = annotation.get("meta", {}).get("origin_path")
    if not origin_path:
        return None

    proprio_path = raw_root / "proprio_stats" / origin_path / "proprio_stats.h5"
    if not proprio_path.exists():
        return None

    with h5py.File(proprio_path, "r") as f:
        state_joint = np.array(f["state/joint/position"], dtype=np.float64)
        state_effector = np.clip(
            (np.array(f["state/effector/position"], dtype=np.float64) - 35.0) / (120.0 - 35.0),
            0.0,
            1.0,
        )
        state_head = np.array(f["state/head/position"], dtype=np.float64)
        state_waist = np.array(f["state/waist/position"], dtype=np.float64)

    states = np.hstack([state_joint, state_effector, state_head, state_waist])
    if states.ndim != 2 or states.shape[1] != 20:
        raise RuntimeError(f"Unexpected recovered AgiBot state shape: {states.shape}")
    return states


def resolve_states(
    annotation: dict,
    checkpoint_metadata: dict,
    total_frames: int,
    raw_agibot_root: Path | None,
) -> tuple[np.ndarray | None, np.ndarray]:
    mean_state = build_mean_state_from_metadata(checkpoint_metadata)
    per_frame_states = None

    if raw_agibot_root is not None:
        per_frame_states = load_states_from_raw_root(raw_agibot_root, annotation)
        if per_frame_states is not None and per_frame_states.shape[0] != total_frames:
            n = min(per_frame_states.shape[0], total_frames)
            per_frame_states = per_frame_states[:n]

    return per_frame_states, mean_state


def build_frame_schedule(
    total_frames: int,
    action_horizon: int,
    obs_frames_per_chunk: int,
    num_chunks: int | None,
) -> list[list[int]]:
    if total_frames <= 0:
        return []

    if obs_frames_per_chunk < 2:
        raise ValueError(f"obs_frames_per_chunk must be >= 2, got {obs_frames_per_chunk}")

    chunks: list[list[int]] = [[0]]
    produced = 0
    chunk_offsets = np.floor(
        np.linspace(0, action_horizon - 1, obs_frames_per_chunk)
    ).astype(np.int64)

    while True:
        if num_chunks is not None and produced >= num_chunks:
            break
        chunk_start = produced * action_horizon
        frame_indices = (chunk_offsets + chunk_start).tolist()
        if frame_indices[-1] >= total_frames:
            break
        chunks.append(frame_indices)
        produced += 1

    return chunks


def make_obs(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    state_history: np.ndarray | None,
    fallback_state: np.ndarray,
    prompt: str,
    language_key: str,
) -> dict:
    obs: dict = {}

    for key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        if len(frame_indices) == 1:
            selected = selected[0]
        obs[key] = selected

    if state_history is not None:
        state_index = min(frame_indices[-1], state_history.shape[0] - 1)
        full_state = state_history[state_index]
    else:
        full_state = fallback_state

    for key, state_slice in STATE_SLICES.items():
        obs[key] = full_state[state_slice].reshape(1, -1).astype(np.float64)

    obs[language_key] = prompt
    return obs


def log_action(step_name: str, result_batch: Batch, dt: float) -> None:
    action_dict = result_batch.act
    print(f"{step_name}: {dt:.2f}s")
    for key in sorted(action_dict.keys()):
        value = action_dict[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        print(
            f"  {key}: shape={value.shape}, "
            f"range=[{value.min():.4f}, {value.max():.4f}]"
        )


def decode_video_prediction(policy: GrootSimPolicy, video_pred: torch.Tensor) -> np.ndarray:
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
    if video.ndim != 4:
        raise ValueError(f"Expected video frames with shape [T, H, W, C], got {video.shape}")
    imageio.mimwrite(output_path, video, fps=fps)


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
    return {key: normalize_action_array(value) for key, value in result_batch.act.items()}


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
        default="/ssd/firefly/Checkpoints/DreamZero-AgiBot",
        help="Path to DreamZero-AgiBot checkpoint directory.",
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
        "--dataset-root",
        default="/ssd/firefly/VAM2/work_dirs/Datasets/EWM_infer_meta/agibot",
        help="Root of EWM AgiBot meta samples.",
    )
    parser.add_argument(
        "--annotations",
        default="annotations_eval_small.json",
        help="Annotation json under dataset root.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Index within the annotations list.",
    )
    parser.add_argument(
        "--raw-agibot-root",
        default="",
        help=(
            "Optional raw AgiBotWorld root. If set and proprio_stats exist, "
            "recover per-frame 20D state from proprio_stats.h5."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional prompt override. Defaults to the sample instruction.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=3,
        help="How many causal action chunks to run after the initial first-frame call.",
    )
    parser.add_argument(
        "--save-video-dir",
        default="outputs/demo_agibot_videos",
        help="Directory to save decoded predicted videos and actions.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="FPS used when saving predicted videos.",
    )
    parser.add_argument(
        "--use-latent-feedback",
        action="store_true",
        help=(
            "Feed the last predicted latent block back into the next causal call, "
            "matching scripts/inference/build_trt_engine.py."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    args = parser.parse_args()

    init_dist()
    if args.device.startswith("cuda"):
        device_index = int(args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_index)

    model_path = Path(args.model_path)
    dataset_root = Path(args.dataset_root)
    annotations_path = dataset_root / args.annotations

    print(f"Using device: {args.device}")
    print(f"Loading checkpoint from: {model_path}")
    print(f"Using tokenizer path: {args.tokenizer_path}")
    print(f"Using Wan checkpoint dir: {args.wan_ckpt_dir}")
    print(f"Reading AgiBot samples from: {annotations_path}")

    annotation = select_episode(annotations_path, args.episode_index)
    camera_frames, prompt = load_episode_assets(dataset_root, annotation)
    if args.prompt:
        prompt = args.prompt
    _ = load_camera_metadata(dataset_root, annotation)

    checkpoint_metadata = load_checkpoint_metadata(model_path)
    total_frames = min(v.shape[0] for v in camera_frames.values())

    raw_root = Path(args.raw_agibot_root) if args.raw_agibot_root else None
    state_history, fallback_state = resolve_states(
        annotation=annotation,
        checkpoint_metadata=checkpoint_metadata,
        total_frames=total_frames,
        raw_agibot_root=raw_root,
    )
    if state_history is not None:
        total_frames = min(total_frames, state_history.shape[0])
        camera_frames = {k: v[:total_frames] for k, v in camera_frames.items()}

    model_config_overrides = [
        f"action_head_cfg.config.diffusion_model_cfg.diffusion_model_pretrained_path={args.wan_ckpt_dir}",
        f"action_head_cfg.config.text_encoder_cfg.text_encoder_pretrained_path={args.wan_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth",
        f"action_head_cfg.config.image_encoder_cfg.image_encoder_pretrained_path={args.wan_ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        f"action_head_cfg.config.vae_cfg.vae_pretrained_path={args.wan_ckpt_dir}/Wan2.1_VAE.pth",
        "action_head_cfg.config.skip_component_loading=true",
    ]

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.AGIBOT,
        model_path=str(model_path),
        tokenizer_path_override=args.tokenizer_path,
        model_config_overrides=model_config_overrides,
        device=args.device,
    )
    print("Policy loaded.")

    action_head = policy.trained_model.action_head
    num_frame_per_block = int(action_head.num_frame_per_block)
    action_horizon = int(action_head.action_horizon)
    language_key = "annotation.detailed_global_instruction_concise"

    frame_schedule = build_frame_schedule(
        total_frames=total_frames,
        action_horizon=action_horizon,
        obs_frames_per_chunk=CAUSAL_OBS_FRAMES,
        num_chunks=args.num_chunks,
    )
    if not frame_schedule:
        raise RuntimeError("No valid AgiBot frame schedule could be built.")

    print(
        f"Episode {args.episode_index}: total_frames={total_frames}, "
        f"num_frame_per_block={num_frame_per_block}, action_horizon={action_horizon}, "
        f"obs_frames_per_chunk={CAUSAL_OBS_FRAMES}, calls={len(frame_schedule)}"
    )
    print(f"Prompt: {prompt}")
    print(
        "State source: "
        + ("raw proprio_stats.h5" if state_history is not None else "checkpoint metadata mean-state fallback")
    )
    print(f"Latent feedback: {args.use_latent_feedback}")

    video_segments: list[np.ndarray] = []
    action_segments: list[dict[str, np.ndarray]] = []
    latent_video: torch.Tensor | None = None

    for i, frame_indices in enumerate(frame_schedule):
        obs = make_obs(
            camera_frames=camera_frames,
            frame_indices=frame_indices,
            state_history=state_history,
            fallback_state=fallback_state,
            prompt=prompt,
            language_key=language_key,
        )
        t0 = time.perf_counter()
        result, video_pred = policy.lazy_joint_forward_causal(
            Batch(obs=obs),
            latent_video=latent_video,
        )
        step_name = "initial" if i == 0 else f"chunk {i}"
        log_action(f"{step_name} {frame_indices}", result, time.perf_counter() - t0)

        decoded_video = decode_video_prediction(policy, video_pred)
        append_video_segment(video_segments, decoded_video[0])
        action_segments.append(extract_action_segment(result))
        if args.use_latent_feedback and video_pred is not None:
            latent_video = video_pred[:, :, -num_frame_per_block:].detach()
        else:
            latent_video = None

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
