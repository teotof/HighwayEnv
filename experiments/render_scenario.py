import argparse
import math
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import imageio.v2 as imageio
import numpy as np
import pygame
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_extractor import KinematicAttentionExtractor  # noqa: F401
from deepset_extractor import DeepSetExtractor  # noqa: F401
from experiments.mappo_core import load_mappo_checkpoint
from experiments.mappo_eval_utils import predict_mappo_actions
from experiments.mappo_utils import (
    mappo_attention_run_name,
    mappo_run_name,
    resolve_mappo_model_path,
)
from experiments.marl_utils import (
    make_shared_world_env,
    predict_shared_actions,
    shared_attention_run_name,
    shared_run_name,
)
from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper
from highway_env.envs.common.graphics import EnvViewer

WRAPPER_MAP = {
    None: None,
    "ShuffleNeighboursObs": ShuffleNeighboursObs,
    "StopGoLeaderWrapper": StopGoLeaderWrapper,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a saved PPO checkpoint for a scenario and optionally save a short video."
    )
    parser.add_argument(
        "--scenario",
        default="cf_v2",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario name from experiments/scenarios.py",
    )
    parser.add_argument(
        "--model-type",
        default="baseline",
        choices=[
            "baseline",
            "attn",
            "deepsets",
            "shared",
            "shared_attn",
            "mappo",
            "mappo_attn",
        ],
        help="Checkpoint family to load.",
    )
    parser.add_argument(
        "--attn-mode",
        default="learned",
        help="Attention mode for attention checkpoints, e.g. learned, uniform, oracle.",
    )
    parser.add_argument(
        "--exp-version",
        default="v2",
        help="Experiment version used in the checkpoint name, e.g. v4, v5, v6.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Checkpoint seed suffix and env reset seed.",
    )
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=2,
        help="Number of controlled vehicles for shared-policy MARL checkpoints.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Override the scenario duration in policy steps. 0 uses the env config duration.",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=30.0,
        help="Sleep rate for human playback. Set 0 to disable sleeping.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=0.0,
        help="If > 0, save an mp4 clip of this many seconds instead of opening a live window.",
    )
    parser.add_argument(
        "--video-dir",
        default="runs/videos",
        help="Directory where mp4 files are saved.",
    )
    parser.add_argument(
        "--name-prefix",
        default="",
        help="Optional prefix for the saved video filename.",
    )
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "rgb_array"],
        help="Render mode when not recording.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional explicit checkpoint path. Overrides model naming logic.",
    )
    parser.add_argument(
        "--observer-agent-index",
        type=int,
        default=0,
        help="Controlled agent index used as the camera center in MARL modes.",
    )
    parser.add_argument(
        "--label-controlled",
        action="store_true",
        help="Draw A0, A1, ... labels for controlled vehicles in MARL renders.",
    )
    parser.add_argument(
        "--fit-controlled",
        action="store_true",
        help="Center the camera on the midpoint of controlled vehicles instead of a single agent.",
    )
    parser.add_argument(
        "--show-attention",
        action="store_true",
        help="Draw an attention summary panel for attention-based models.",
    )
    parser.add_argument(
        "--split-controlled-view",
        action="store_true",
        help="Render one top view per controlled agent side-by-side, with an optional bottom summary strip.",
    )
    parser.add_argument(
        "--split-gap",
        type=int,
        default=12,
        help="Gap in pixels between split-screen agent views and between the top row and the bottom strip.",
    )
    parser.add_argument(
        "--attn-top-k",
        type=int,
        default=3,
        help="How many neighbour slots per agent to list in the attention panel.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used when loading checkpoints, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=0,
        help="Optional render width override in pixels. 0 keeps the scenario/default value.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=0,
        help="Optional render height override in pixels. 0 keeps the scenario/default value.",
    )
    parser.add_argument(
        "--render-scaling",
        type=float,
        default=0.0,
        help="Optional render scaling override. 0 keeps the scenario/default value.",
    )
    parser.add_argument(
        "--random-agent",
        action="store_true",
        help="Use random actions instead of loading a PPO checkpoint.",
    )
    return parser.parse_args()


def resolve_model_path(args) -> Path:
    if args.model_path:
        path = Path(args.model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Model path does not exist: {path}")

    models_dir = Path("runs/models")
    if args.model_type == "baseline":
        candidates = [
            models_dir / f"ppo_baseline_{args.scenario}_{args.exp_version}_seed{args.seed}.zip"
        ]
    elif args.model_type == "deepsets":
        candidates = [
            models_dir / f"ppo_deepsets_{args.scenario}_{args.exp_version}_seed{args.seed}.zip"
        ]
    elif args.model_type == "shared":
        candidates = [
            models_dir
            / f"{shared_run_name(args.scenario, args.exp_version, args.controlled_vehicles, args.seed)}.zip"
        ]
    elif args.model_type == "shared_attn":
        candidates = [
            models_dir
            / (
                f"{shared_attention_run_name(args.scenario, args.exp_version, args.controlled_vehicles, args.attn_mode, args.seed)}.zip"
            )
        ]
    elif args.model_type == "mappo":
        candidates = [
            resolve_mappo_model_path(
                models_dir,
                args.scenario,
                args.exp_version,
                args.controlled_vehicles,
                args.seed,
                model_kind="baseline",
            )
        ]
    elif args.model_type == "mappo_attn":
        candidates = [
            resolve_mappo_model_path(
                models_dir,
                args.scenario,
                args.exp_version,
                args.controlled_vehicles,
                args.seed,
                model_kind="attn",
                attn_mode=args.attn_mode,
            )
        ]
    else:
        candidates = [
            models_dir
            / f"ppo_attn_{args.attn_mode}_{args.scenario}_{args.exp_version}_seed{args.seed}.zip",
            models_dir / f"ppo_attn_{args.scenario}_{args.exp_version}_seed{args.seed}.zip",
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No checkpoint found. Tried:\n{tried}")


def make_env(args, render_mode: str):
    if args.model_type in {"shared", "shared_attn", "mappo", "mappo_attn"}:
        env = make_shared_world_env(
            scenario_name=args.scenario,
            controlled_vehicles=args.controlled_vehicles,
            render_mode=render_mode,
        )
        apply_render_overrides(env, args)
        return env

    scenario = SCENARIOS[args.scenario]
    env = gym.make(
        scenario["env_id"],
        config=scenario["config"],
        render_mode=render_mode,
    )

    wrapper_name = scenario.get("wrapper", None)
    wrapper_kwargs = scenario.get("wrapper_kwargs", {})
    wrapper_class = WRAPPER_MAP.get(wrapper_name)
    if wrapper_class is not None:
        env = wrapper_class(env, **wrapper_kwargs)

    apply_render_overrides(env, args)
    return env


def apply_render_overrides(env, args):
    base_env = env.unwrapped
    if args.screen_width > 0:
        base_env.config["screen_width"] = int(args.screen_width)
    if args.screen_height > 0:
        base_env.config["screen_height"] = int(args.screen_height)
    if args.render_scaling > 0:
        base_env.config["scaling"] = float(args.render_scaling)
    if args.split_controlled_view:
        base_env.config["offscreen_rendering"] = True
        base_env.config["render_agent"] = False


def make_video_prefix(args) -> str:
    if args.name_prefix:
        return args.name_prefix
    if args.model_type == "baseline":
        return f"{args.scenario}_{args.exp_version}_seed{args.seed}"
    if args.model_type == "deepsets":
        return f"deepsets_{args.scenario}_{args.exp_version}_seed{args.seed}"
    if args.model_type == "shared":
        return (
            f"shared_{args.scenario}_{args.exp_version}_"
            f"cv{args.controlled_vehicles}_seed{args.seed}"
        )
    if args.model_type == "shared_attn":
        return (
            f"shared_attn_{args.attn_mode}_{args.scenario}_{args.exp_version}_"
            f"cv{args.controlled_vehicles}_seed{args.seed}"
        )
    if args.model_type == "mappo":
        return mappo_run_name(
            args.scenario,
            args.exp_version,
            args.controlled_vehicles,
            args.seed,
        )
    if args.model_type == "mappo_attn":
        return mappo_attention_run_name(
            args.scenario,
            args.exp_version,
            args.controlled_vehicles,
            args.attn_mode,
            args.seed,
        )
    return f"{args.attn_mode}_{args.scenario}_{args.exp_version}_seed{args.seed}"


def maybe_wrap_video(env, args):
    if args.split_controlled_view:
        return env, None, None
    if args.record_seconds <= 0:
        return env, None, None

    video_dir = Path(args.video_dir)
    name_prefix = make_video_prefix(args)

    try:
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: episode_id == 0,
            video_length=1,
            name_prefix=name_prefix,
        )
    except DependencyNotInstalled as exc:
        raise RuntimeError(
            'Video recording requires MoviePy. Install it with `.\\.venv\\Scripts\\python.exe -m pip install "gymnasium[other]" moviepy`.'
        ) from exc

    env.unwrapped.set_record_video_wrapper(env)
    env.video_length = max(1, math.ceil(args.record_seconds * env.frames_per_sec))
    return env, video_dir, name_prefix


def load_policy(args):
    if args.random_agent:
        return None, None

    model_path = resolve_model_path(args)
    if args.model_type in {"mappo", "mappo_attn"}:
        model, _ = load_mappo_checkpoint(str(model_path), device=args.device)
        return model, model_path

    model = PPO.load(str(model_path), device=args.device)
    return model, model_path


def find_latest_video(video_dir: Path | None, prefix: str | None):
    if not video_dir or not prefix or not video_dir.exists():
        return None
    videos = sorted(video_dir.glob(f"{prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
    return videos[-1] if videos else None


def supports_attention_overlay(args) -> bool:
    return args.model_type in {"attn", "shared_attn", "mappo_attn"}


def extract_attention_matrix(model, args):
    attn = None
    if args.model_type == "mappo_attn":
        attn = getattr(model, "last_attention", None)
    elif args.model_type in {"attn", "shared_attn"}:
        extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
        attn = getattr(extractor, "last_attention", None)

    if attn is None:
        return None
    if hasattr(attn, "detach"):
        attn = attn.detach().cpu().numpy()
    attn = np.asarray(attn, dtype=np.float32)
    if attn.ndim == 1:
        attn = attn[None, :]
    return attn


def draw_text(surface, text, x, y, font, color=(255, 255, 255), bg=(0, 0, 0, 160)):
    rendered = font.render(text, True, color)
    if bg is not None:
        bg_surface = pygame.Surface(
            (rendered.get_width() + 6, rendered.get_height() + 4),
            flags=pygame.SRCALPHA,
        )
        bg_surface.fill(bg)
        surface.blit(bg_surface, (x - 3, y - 2))
    surface.blit(rendered, (x, y))


def update_viewer_camera(env, args, *, observer_agent_index=None, fit_controlled=None):
    base_env = env.unwrapped
    viewer = getattr(base_env, "viewer", None)
    vehicles = list(getattr(base_env, "controlled_vehicles", []) or [])
    if viewer is None or not vehicles:
        return

    use_fit_controlled = args.fit_controlled if fit_controlled is None else bool(fit_controlled)
    if observer_agent_index is None:
        observer_agent_index = args.observer_agent_index

    base_centering = list(base_env.config.get("centering_position", [0.3, 0.5]))
    base_scaling = float(
        base_env.config.get("scaling", viewer.sim_surface.INITIAL_SCALING)
    )

    if use_fit_controlled and len(vehicles) > 1:
        positions = np.stack(
            [np.asarray(vehicle.position, dtype=np.float32) for vehicle in vehicles],
            axis=0,
        )
        midpoint = positions.mean(axis=0)
        span = np.maximum(positions.max(axis=0) - positions.min(axis=0), 0.0)
        margin = np.maximum(
            np.array([40.0, 10.0], dtype=np.float32),
            0.10 * span + np.array([8.0, 4.0], dtype=np.float32),
        )
        world_size = np.maximum(span + 2.0 * margin, np.array([35.0, 12.0], dtype=np.float32))
        fit_scale_x = viewer.sim_surface.get_width() / float(world_size[0])
        fit_scale_y = viewer.sim_surface.get_height() / float(world_size[1])
        viewer.sim_surface.scaling = max(1.0, min(base_scaling, fit_scale_x, fit_scale_y))
        viewer.sim_surface.centering_position = [0.5, 0.5]
        viewer.observer_vehicle = SimpleNamespace(position=midpoint)
    else:
        observer_idx = max(0, min(int(observer_agent_index), len(vehicles) - 1))
        viewer.observer_vehicle = vehicles[observer_idx]
        viewer.sim_surface.scaling = base_scaling
        viewer.sim_surface.centering_position = base_centering


def draw_attention_summary(surface, args, obs, attn, vehicles, *, header_line):
    label_colors = [
        (80, 220, 120),
        (255, 180, 60),
        (120, 190, 255),
        (255, 120, 180),
    ]
    small_font = pygame.font.Font(None, 16)

    surface.fill((18, 18, 18))
    draw_text(surface, header_line, 12, 10, small_font, bg=None)

    if attn is None or obs is None:
        return

    obs_batch = np.asarray(obs, dtype=np.float32)
    if obs_batch.ndim == 2:
        obs_batch = obs_batch[None, ...]

    agents = min(attn.shape[0], obs_batch.shape[0])
    panel_width = surface.get_width()
    panel_height = surface.get_height()
    column_gap = 16
    column_count = max(1, agents)
    column_width = max(180, (panel_width - 20 - column_gap * (column_count - 1)) // column_count)
    block_top = 32

    for agent_idx in range(agents):
        agent_attn = attn[agent_idx]
        neighbours = obs_batch[agent_idx, 1:, :]
        valid = neighbours[:, 0] > 0.5
        valid_indices = np.flatnonzero(valid)
        x0 = 12 + agent_idx * (column_width + column_gap)
        y0 = block_top
        block_height = panel_height - block_top - 10
        block_surface = pygame.Surface((column_width, max(1, block_height)), flags=pygame.SRCALPHA)
        block_surface.fill((35, 35, 35, 220))
        surface.blit(block_surface, (x0, y0))

        title_color = label_colors[agent_idx % len(label_colors)]
        entropy = (
            float(-(agent_attn[valid] * np.log(np.clip(agent_attn[valid], 1e-8, 1.0))).sum())
            if valid_indices.size > 0
            else 0.0
        )
        desired_top_k = max(1, min(args.attn_top_k, valid_indices.size)) if valid_indices.size > 0 else 0
        top_indices = (
            valid_indices[np.argsort(agent_attn[valid_indices])[::-1][:desired_top_k]]
            if desired_top_k > 0
            else np.array([], dtype=np.int64)
        )

        body_lines = []
        if valid_indices.size == 0:
            body_lines.append("no visible neighbours")
        else:
            for slot_idx in top_indices:
                dx = float(neighbours[slot_idx, 1])
                vx = float(neighbours[slot_idx, 3])
                body_lines.append(
                    f"s{slot_idx+1}: w={float(agent_attn[slot_idx]):.2f} dx={dx:+.1f} vx={vx:+.1f}"
                )

        padding_y = 8
        available_height = max(24, block_height - 2 * padding_y)
        title_font = None
        body_font = None
        line_height = None
        for font_size in [18, 16, 15, 14, 13, 12]:
            candidate_title = pygame.font.Font(None, font_size + 2)
            candidate_body = pygame.font.Font(None, font_size)
            candidate_line_height = max(candidate_body.get_linesize(), font_size)
            needed_height = candidate_title.get_linesize() + 4 + max(len(body_lines), 1) * candidate_line_height
            if needed_height <= available_height:
                title_font = candidate_title
                body_font = candidate_body
                line_height = candidate_line_height
                break

        if title_font is None or body_font is None or line_height is None:
            title_font = pygame.font.Font(None, 14)
            body_font = pygame.font.Font(None, 12)
            line_height = max(body_font.get_linesize(), 12)
            remaining_height = max(0, available_height - title_font.get_linesize() - 4)
            max_body_lines = max(1, remaining_height // max(line_height, 1))
            if len(body_lines) > max_body_lines:
                visible_lines = body_lines[: max(0, max_body_lines - 1)]
                hidden = len(body_lines) - len(visible_lines)
                body_lines = visible_lines + [f"... {hidden} more"]

        draw_text(surface, f"A{agent_idx}  H={entropy:.2f}", x0 + 8, y0 + padding_y, title_font, color=title_color, bg=None)
        y_cursor = y0 + padding_y + title_font.get_linesize() + 4
        for line in body_lines:
            draw_text(surface, line, x0 + 8, y_cursor, body_font, bg=None)
            y_cursor += line_height


def build_overlay_callback(args, env, overlay_state):
    label_colors = [
        (80, 220, 120),
        (255, 180, 60),
        (120, 190, 255),
        (255, 120, 180),
    ]

    def callback(agent_surface, sim_surface):
        base_env = env.unwrapped
        vehicles = list(getattr(base_env, "controlled_vehicles", []) or [])
        if vehicles:
            update_viewer_camera(env, args)

        font = pygame.font.Font(None, 20)
        small_font = pygame.font.Font(None, 16)

        if args.label_controlled and vehicles:
            for idx, vehicle in enumerate(vehicles):
                vehicle.color = label_colors[idx % len(label_colors)]
                px, py = sim_surface.pos2pix(float(vehicle.position[0]), float(vehicle.position[1]))
                color = label_colors[idx % len(label_colors)]
                px = int(px)
                py = int(py)
                pygame.draw.circle(sim_surface, color, (px, py), 12, width=2)
                pygame.draw.circle(sim_surface, (255, 255, 255), (px, py), 2)
                label_x = min(max(6, px + 10), sim_surface.get_width() - 36)
                label_y = min(max(6, py - 18), sim_surface.get_height() - 22)
                draw_text(sim_surface, f"A{idx}", label_x, label_y, font, color=color)

        if agent_surface is None:
            return

        attn = overlay_state.get("attention")
        obs = overlay_state.get("obs")
        if not args.show_attention or attn is None or obs is None:
            agent_surface.fill((18, 18, 18))
            if vehicles:
                summary = [f"Controlled vehicles: {len(vehicles)}"]
                if args.fit_controlled and len(vehicles) > 1:
                    summary.append("Camera: midpoint of controlled vehicles")
                else:
                    observer_idx = max(0, min(args.observer_agent_index, len(vehicles) - 1))
                    summary.append(f"Camera: A{observer_idx}")
                for idx, line in enumerate(summary):
                    draw_text(agent_surface, line, 14, 14 + idx * 20, font, bg=None)
            return

        header_line = f"Attention: {args.attn_mode}"
        if vehicles:
            if args.fit_controlled and len(vehicles) > 1:
                header_line += f" | Controlled: {len(vehicles)} | Camera: midpoint"
            else:
                observer_idx = max(0, min(args.observer_agent_index, len(vehicles) - 1))
                header_line += f" | Controlled: {len(vehicles)} | Camera: A{observer_idx}"
        draw_attention_summary(agent_surface, args, obs, attn, vehicles, header_line=header_line)

    return callback


def surface_to_image(surface):
    data = pygame.surfarray.array3d(surface)
    return np.moveaxis(data, 0, 1)


def ensure_even_frame(frame):
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def build_split_bottom_panel(args, overlay_state, vehicles, top_width, per_view_height):
    if not (args.show_attention and overlay_state.get("attention") is not None and overlay_state.get("obs") is not None):
        return None

    panel_height = max(140, min(220, per_view_height // 2 if per_view_height > 0 else 160))
    surface = pygame.Surface((top_width, panel_height))
    header_line = f"Attention: {args.attn_mode} | Split view across {len(vehicles)} agents"
    draw_attention_summary(
        surface,
        args,
        overlay_state.get("obs"),
        overlay_state.get("attention"),
        vehicles,
        header_line=header_line,
    )
    return surface_to_image(surface)


def compose_split_frame(env, args, overlay_state):
    base_env = env.unwrapped
    vehicles = list(getattr(base_env, "controlled_vehicles", []) or [])
    if not vehicles:
        return env.render()

    if getattr(base_env, "viewer", None) is None:
        env.render()

    frames = []
    for agent_idx in range(len(vehicles)):
        update_viewer_camera(env, args, observer_agent_index=agent_idx, fit_controlled=False)
        frame = env.render()
        frames.append(np.asarray(frame, dtype=np.uint8))

    gap = max(0, int(args.split_gap))
    frame_h = frames[0].shape[0]
    frame_w = frames[0].shape[1]
    top_width = len(frames) * frame_w + max(0, len(frames) - 1) * gap
    top_row = np.zeros((frame_h, top_width, 3), dtype=np.uint8)
    x_offset = 0
    for frame in frames:
        top_row[:, x_offset : x_offset + frame_w, :] = frame
        x_offset += frame_w + gap

    if gap > 0:
        font = pygame.font.Font(None, 24)
        colors = [
            (80, 220, 120),
            (255, 180, 60),
            (120, 190, 255),
            (255, 120, 180),
        ]
        overlay = pygame.Surface((top_width, frame_h), flags=pygame.SRCALPHA)
        x_offset = 0
        for agent_idx in range(len(frames)):
            label = f"A{agent_idx}"
            draw_text(
                overlay,
                label,
                x_offset + 10,
                10,
                font,
                color=colors[agent_idx % len(colors)],
                bg=(0, 0, 0, 170),
            )
            x_offset += frame_w + gap
        overlay_img = surface_to_image(overlay)
        alpha_mask = np.any(overlay_img != 0, axis=2)
        top_row[alpha_mask] = overlay_img[alpha_mask]

    bottom_panel = build_split_bottom_panel(
        args,
        overlay_state,
        vehicles,
        top_row.shape[1],
        top_row.shape[0],
    )
    if bottom_panel is not None:
        if gap > 0:
            separator = np.zeros((gap, top_row.shape[1], 3), dtype=np.uint8)
            return np.concatenate([top_row, separator, bottom_panel], axis=0)
        return np.concatenate([top_row, bottom_panel], axis=0)
    return top_row


class SplitScreenRecorder:
    def __init__(self, env, args, overlay_state, video_path: Path, fps: float, max_frames: int):
        self.env = env
        self.args = args
        self.overlay_state = overlay_state
        self.video_path = video_path
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(video_path), fps=fps, macro_block_size=1)
        self.frames_per_sec = fps
        self.max_frames = max_frames
        self.frames_written = 0
        self.finished = False

    def _capture_frame(self):
        if self.finished:
            return
        frame = compose_split_frame(self.env, self.args, self.overlay_state)
        frame = ensure_even_frame(np.asarray(frame, dtype=np.uint8))
        self.writer.append_data(frame)
        self.frames_written += 1
        if self.frames_written >= self.max_frames:
            self.finished = True

    def close(self):
        self.writer.close()


def make_split_video_path(args):
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir / f"{make_video_prefix(args)}-episode-0.mp4"


def show_split_frame(frame, window_state):
    if window_state.get("screen") is None:
        pygame.init()
        pygame.display.set_caption("Highway-env split")
        window_state["screen"] = pygame.display.set_mode((frame.shape[1], frame.shape[0]))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            window_state["closed"] = True

    if window_state.get("closed"):
        return

    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    window_state["screen"].blit(surface, (0, 0))
    pygame.display.flip()


def main():
    args = parse_args()
    if args.split_controlled_view:
        if args.model_type not in {"shared", "shared_attn", "mappo", "mappo_attn"}:
            raise ValueError("--split-controlled-view only supports MARL/shared model types.")
        if args.controlled_vehicles < 2:
            raise ValueError("--split-controlled-view requires --controlled-vehicles >= 2.")
    if args.show_attention and not supports_attention_overlay(args):
        print(
            f"Attention overlay ignored for model type '{args.model_type}'. "
            "Use attn, shared_attn, or mappo_attn.",
            flush=True,
        )
        args.show_attention = False

    model, model_path = load_policy(args)

    recording = args.record_seconds > 0
    if args.split_controlled_view:
        render_mode = "rgb_array"
    else:
        render_mode = "rgb_array" if recording else args.render_mode
    env = make_env(args, render_mode=render_mode)
    overlay_state = {"attention": None, "obs": None}
    if args.show_attention or args.label_controlled:
        overlay_callback = build_overlay_callback(args, env, overlay_state)
        EnvViewer.agent_display = overlay_callback
        if getattr(env.unwrapped, "viewer", None) is not None:
            if getattr(env.unwrapped.viewer, "agent_surface", None) is None:
                env.unwrapped.viewer.extend_display()
            env.unwrapped.viewer.set_agent_display(overlay_callback)
            update_viewer_camera(env, args)
    env, video_dir, video_prefix = maybe_wrap_video(env, args)

    if args.max_steps > 0:
        max_steps = args.max_steps
    else:
        cfg = env.unwrapped.config
        max_steps = max(1, int(round(float(cfg.get("duration", 40)) * float(cfg.get("policy_frequency", 1)))))

    if model_path is not None:
        print(f"Loaded model: {model_path}")
    else:
        print("Using random actions.")

    print(f"Scenario: {args.scenario}")
    print(f"Render mode: {render_mode}")
    print(f"Max steps per episode: {max_steps}")
    if args.model_type in {"shared", "shared_attn", "mappo", "mappo_attn"}:
        print(f"Controlled vehicles: {args.controlled_vehicles}")
    if args.split_controlled_view:
        print("Split controlled view: enabled")

    split_recorder = None
    split_video_path = None
    split_window_state = {"screen": None, "closed": False}
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            overlay_state["obs"] = np.asarray(obs, dtype=np.float32)
            update_viewer_camera(env, args)
            if model is not None and args.show_attention:
                if args.model_type in {"shared", "shared_attn"}:
                    _ = predict_shared_actions(model, obs, deterministic=True)
                elif args.model_type in {"mappo", "mappo_attn"}:
                    _ = predict_mappo_actions(model, obs, deterministic=True)
                else:
                    _ = model.predict(obs, deterministic=True)
                overlay_state["attention"] = extract_attention_matrix(model, args)
            terminated = truncated = False
            step = 0
            clip_finished = False

            if args.split_controlled_view and recording:
                split_video_path = make_split_video_path(args)
                render_fps = float(env.metadata.get("render_fps", env.unwrapped.config.get("simulation_frequency", 15)))
                max_frames = max(1, math.ceil(args.record_seconds * render_fps))
                split_recorder = SplitScreenRecorder(
                    env,
                    args,
                    overlay_state,
                    split_video_path,
                    fps=render_fps,
                    max_frames=max_frames,
                )
                env.unwrapped.set_record_video_wrapper(split_recorder)
                split_recorder._capture_frame()
            elif args.split_controlled_view and args.render_mode == "human":
                initial_frame = compose_split_frame(env, args, overlay_state)
                show_split_frame(initial_frame, split_window_state)

            while not (terminated or truncated) and step < max_steps:
                update_viewer_camera(env, args)
                if model is None:
                    action = env.action_space.sample()
                elif args.model_type in {"shared", "shared_attn"}:
                    action = predict_shared_actions(model, obs, deterministic=True)
                elif args.model_type in {"mappo", "mappo_attn"}:
                    action = predict_mappo_actions(model, obs, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)

                if model is not None and args.show_attention:
                    overlay_state["obs"] = np.asarray(obs, dtype=np.float32)
                    overlay_state["attention"] = extract_attention_matrix(model, args)

                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                overlay_state["obs"] = np.asarray(obs, dtype=np.float32)

                if args.split_controlled_view and recording:
                    if split_recorder is not None:
                        split_recorder._capture_frame()
                        if split_recorder.finished:
                            clip_finished = True
                            break
                elif args.split_controlled_view and args.render_mode == "human":
                    frame = compose_split_frame(env, args, overlay_state)
                    show_split_frame(frame, split_window_state)
                    if split_window_state.get("closed"):
                        clip_finished = True
                        terminated = True
                        truncated = True
                        break
                    if args.playback_fps > 0:
                        time.sleep(1.0 / args.playback_fps)

                if render_mode == "human" and not args.split_controlled_view and args.playback_fps > 0:
                    time.sleep(1.0 / args.playback_fps)

                if recording and not args.split_controlled_view and not env.recording:
                    clip_finished = True
                    break

            print(f"Episode {episode + 1}/{args.episodes} finished after {step} steps.")

            if split_recorder is not None:
                split_recorder.close()
                split_recorder = None
                env.unwrapped._record_video_wrapper = None

            if clip_finished:
                break
    finally:
        if split_recorder is not None:
            split_recorder.close()
            env.unwrapped._record_video_wrapper = None
        if split_window_state.get("screen") is not None:
            pygame.display.quit()
        EnvViewer.agent_display = None
        env.close()

    if args.split_controlled_view and recording:
        if split_video_path is not None and split_video_path.exists():
            print(f"Saved video: {split_video_path}")
    elif recording:
        latest_video = find_latest_video(video_dir, video_prefix)
        if latest_video is not None:
            print(f"Saved video: {latest_video}")


if __name__ == "__main__":
    main()
