import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.marl_utils import make_shared_world_env, predict_shared_actions, shared_run_name


def parse_seeds(raw: str):
    if not raw:
        return [0, 1, 2]
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    return seeds if seeds else [0, 1, 2]


TRAIN_SCENARIO_NAME = os.getenv("TRAIN_SCENARIO_NAME", os.getenv("SCENARIO_NAME", "cf_v2"))
EVAL_SCENARIO_NAME = os.getenv("EVAL_SCENARIO_NAME", TRAIN_SCENARIO_NAME)
EXP_VERSION = os.getenv("EXP_VERSION", "v10")
CONTROLLED_VEHICLES = int(os.getenv("CONTROLLED_VEHICLES", "2"))
SEEDS = parse_seeds(os.getenv("SEEDS", "0,1,2"))
N_EVAL_EPISODES = int(os.getenv("N_EVAL_EPISODES", "50"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "0"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "runs/models"))


def auto_max_steps(env) -> int:
    cfg = env.unwrapped.config
    duration_s = float(cfg.get("duration", 40))
    policy_hz = float(cfg.get("policy_frequency", 1))
    return max(1, int(round(duration_s * policy_hz)))


def eval_one(model_path: str, seed: int):
    env = make_shared_world_env(
        scenario_name=EVAL_SCENARIO_NAME,
        controlled_vehicles=CONTROLLED_VEHICLES,
    )
    model = PPO.load(model_path, device="cpu")
    max_steps = MAX_STEPS if MAX_STEPS > 0 else auto_max_steps(env)

    team_returns = []
    agent_return_means = []
    lengths = []
    world_crashes = 0
    agent_crashes = 0
    world_successes = 0

    for episode in range(N_EVAL_EPISODES):
        obs, info = env.reset(seed=seed * 1000 + episode)
        terminated = truncated = False
        team_return = 0.0
        agent_returns = np.zeros((CONTROLLED_VEHICLES,), dtype=np.float32)
        episode_steps = 0

        while not (terminated or truncated) and episode_steps < max_steps:
            action = predict_shared_actions(model, obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            team_return += float(reward)

            if "agents_rewards" in info:
                agent_returns += np.asarray(info["agents_rewards"], dtype=np.float32)
            else:
                agent_returns += float(reward)

            episode_steps += 1

        team_returns.append(team_return)
        agent_return_means.append(float(agent_returns.mean()))
        lengths.append(episode_steps)

        crashed = [
            bool(getattr(vehicle, "crashed", False))
            for vehicle in env.unwrapped.controlled_vehicles
        ]
        world_crashes += int(any(crashed))
        agent_crashes += int(sum(crashed))

        if bool(info.get("is_success", False)):
            world_successes += 1

        if episode % 5 == 0:
            print(
                f"[{os.path.basename(model_path)}] seed={seed} "
                f"ep={episode}/{N_EVAL_EPISODES}",
                flush=True,
            )

    env.close()

    return {
        "mean_team_return": float(np.mean(team_returns)),
        "std_team_return": float(np.std(team_returns)),
        "mean_agent_return": float(np.mean(agent_return_means)),
        "mean_len": float(np.mean(lengths)),
        "world_crash_rate": float(world_crashes / N_EVAL_EPISODES),
        "agent_crash_rate": float(
            agent_crashes / max(N_EVAL_EPISODES * CONTROLLED_VEHICLES, 1)
        ),
        "world_success_rate": float(world_successes / N_EVAL_EPISODES),
    }


if __name__ == "__main__":
    print(f"Train scenario: {TRAIN_SCENARIO_NAME}")
    print(f"Eval scenario: {EVAL_SCENARIO_NAME}")
    print(f"EXP_VERSION={EXP_VERSION}")
    print(f"CONTROLLED_VEHICLES={CONTROLLED_VEHICLES}")
    print(f"N_EVAL_EPISODES={N_EVAL_EPISODES}")

    results = []
    for seed in SEEDS:
        model_path = (
            MODEL_DIR
            / f"{shared_run_name(TRAIN_SCENARIO_NAME, EXP_VERSION, CONTROLLED_VEHICLES, seed)}.zip"
        )
        results.append(eval_one(str(model_path), seed))

    for metric in [
        "mean_team_return",
        "mean_agent_return",
        "world_crash_rate",
        "agent_crash_rate",
        "world_success_rate",
        "mean_len",
    ]:
        values = [result[metric] for result in results]
        print(f"{metric}: {np.mean(values):.3f} +/- {np.std(values):.3f}")
