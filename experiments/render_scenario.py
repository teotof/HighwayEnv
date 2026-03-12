import argparse
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_extractor import KinematicAttentionExtractor  # noqa: F401
from experiments.scenarios import SCENARIOS
from experiments.wrappers import ShuffleNeighboursObs, StopGoLeaderWrapper

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
        choices=["baseline", "attn"],
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

    return env


def make_video_prefix(args) -> str:
    if args.name_prefix:
        return args.name_prefix
    if args.model_type == "baseline":
        return f"{args.scenario}_{args.exp_version}_seed{args.seed}"
    return f"{args.attn_mode}_{args.scenario}_{args.exp_version}_seed{args.seed}"


def maybe_wrap_video(env, args):
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
    model = PPO.load(str(model_path), device="cpu")
    return model, model_path


def find_latest_video(video_dir: Path | None, prefix: str | None):
    if not video_dir or not prefix or not video_dir.exists():
        return None
    videos = sorted(video_dir.glob(f"{prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
    return videos[-1] if videos else None


def main():
    args = parse_args()
    model, model_path = load_policy(args)

    recording = args.record_seconds > 0
    render_mode = "rgb_array" if recording else args.render_mode
    env = make_env(args, render_mode=render_mode)
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

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            step = 0
            clip_finished = False

            while not (terminated or truncated) and step < max_steps:
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                step += 1

                if render_mode == "human" and args.playback_fps > 0:
                    time.sleep(1.0 / args.playback_fps)

                if recording and not env.recording:
                    clip_finished = True
                    break

            print(f"Episode {episode + 1}/{args.episodes} finished after {step} steps.")

            if clip_finished:
                break
    finally:
        env.close()

    if recording:
        latest_video = find_latest_video(video_dir, video_prefix)
        if latest_video is not None:
            print(f"Saved video: {latest_video}")


if __name__ == "__main__":
    main()
