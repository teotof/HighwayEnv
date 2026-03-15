import os
import sys
from pathlib import Path

from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.marl_utils import (
    make_attention_policy_kwargs,
    make_shared_world_env,
    make_shared_world_env_factory,
    shared_attention_run_name,
)
from experiments.shared_policy_vec_env import SharedPolicyVecEnv


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


SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
EXP_VERSION = os.getenv("EXP_VERSION", "v10")
SEEDS = parse_seeds(os.getenv("SEEDS", "0,1,2"))
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
CONTROLLED_VEHICLES = int(os.getenv("CONTROLLED_VEHICLES", "2"))
N_WORLDS = int(os.getenv("N_WORLDS", "4"))
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "runs/models"))
MONITOR_ROOT = Path(os.getenv("MONITOR_ROOT", "runs/monitor"))
TB_DIR = os.getenv("TB_DIR", "runs/tb")

DEFAULT_VEC_MODE = "sync" if os.name == "nt" else "async"
VEC_MODE = os.getenv("VEC_MODE", DEFAULT_VEC_MODE).strip().lower()

TARGET_ROLLOUT_SIZE = int(os.getenv("TARGET_ROLLOUT_SIZE", "8192"))
PPO_N_STEPS = int(
    os.getenv(
        "PPO_N_STEPS",
        str(max(128, TARGET_ROLLOUT_SIZE // max(N_WORLDS * CONTROLLED_VEHICLES, 1))),
    )
)

ATTN_MODE = os.getenv("ATTN_MODE", "learned").strip().lower()
X_INDEX = int(os.getenv("X_INDEX", "1"))
VX_INDEX = int(os.getenv("VX_INDEX", "3"))
ORACLE_RULE = os.getenv("ORACLE_RULE", "leader_x").strip().lower()
TTC_EPS = float(os.getenv("TTC_EPS", "1e-6"))


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MONITOR_ROOT.mkdir(parents=True, exist_ok=True)

    policy_kwargs = make_attention_policy_kwargs(
        attn_mode=ATTN_MODE,
        x_index=X_INDEX,
        vx_index=VX_INDEX,
        oracle_rule=ORACLE_RULE,
        ttc_eps=TTC_EPS,
    )

    for seed in SEEDS:
        run_name = shared_attention_run_name(
            SCENARIO_NAME,
            EXP_VERSION,
            CONTROLLED_VEHICLES,
            ATTN_MODE,
            seed,
        )

        monitor_dir = MONITOR_ROOT / run_name
        monitor_dir.mkdir(parents=True, exist_ok=True)

        test_env = make_shared_world_env(
            scenario_name=SCENARIO_NAME,
            controlled_vehicles=CONTROLLED_VEHICLES,
        )
        obs, _ = test_env.reset(seed=seed)
        print(
            run_name,
            "agents:",
            len(obs),
            "agent obs shape:",
            obs[0].shape,
            "joint action space:",
            test_env.action_space,
        )
        test_env.close()

        env_fns = [
            make_shared_world_env_factory(
                scenario_name=SCENARIO_NAME,
                controlled_vehicles=CONTROLLED_VEHICLES,
                monitor_path=str(monitor_dir / f"{world_index}"),
            )
            for world_index in range(N_WORLDS)
        ]
        env = SharedPolicyVecEnv(
            env_fns,
            num_agents=CONTROLLED_VEHICLES,
            vectorization_mode=VEC_MODE,
        )

        print(
            f"{run_name}: worlds={N_WORLDS}, agents_per_world={CONTROLLED_VEHICLES}, "
            f"ppo_slots={env.num_envs}, vec_mode={VEC_MODE}, n_steps={PPO_N_STEPS}, "
            f"attn_mode={ATTN_MODE}, oracle_rule={ORACLE_RULE}",
            flush=True,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=TB_DIR,
            device=DEVICE,
            seed=seed,
            n_steps=PPO_N_STEPS,
            policy_kwargs=policy_kwargs,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_name)
        model.save(str(MODEL_DIR / run_name))
        env.close()
