import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mappo_core import (  # noqa: E402
    MAPPOConfig,
    MAPPOPolicy,
    explained_variance,
    save_mappo_checkpoint,
)
from experiments.mappo_utils import (  # noqa: E402
    build_centralized_inputs,
    extract_slot_rewards,
    mappo_run_name,
    parse_seeds,
    world_metrics_from_slots,
)
from experiments.marl_utils import (  # noqa: E402
    make_shared_world_env,
    make_shared_world_env_factory,
)
from experiments.shared_policy_vec_env import SharedPolicyVecEnv  # noqa: E402


SCENARIO_NAME = os.getenv("SCENARIO_NAME", "cf_v2")
EXP_VERSION = os.getenv("EXP_VERSION", "v11")
SEEDS = parse_seeds(os.getenv("SEEDS", "0,1,2"))
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
CONTROLLED_VEHICLES = int(os.getenv("CONTROLLED_VEHICLES", "2"))
N_WORLDS = int(os.getenv("N_WORLDS", "4"))
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "runs/models"))
MONITOR_ROOT = Path(os.getenv("MONITOR_ROOT", "runs/monitor"))
TB_DIR = Path(os.getenv("TB_DIR", "runs/tb"))
REWARD_MODE = os.getenv("REWARD_MODE", "team").strip().lower()
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "128"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-4"))
GAMMA = float(os.getenv("GAMMA", "0.99"))
GAE_LAMBDA = float(os.getenv("GAE_LAMBDA", "0.95"))
CLIP_RANGE = float(os.getenv("CLIP_RANGE", "0.2"))
ENT_COEF = float(os.getenv("ENT_COEF", "0.0"))
VF_COEF = float(os.getenv("VF_COEF", "0.5"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "0.5"))
UPDATE_EPOCHS = int(os.getenv("UPDATE_EPOCHS", "10"))
NUM_MINIBATCHES = int(os.getenv("NUM_MINIBATCHES", "4"))
TARGET_KL_RAW = os.getenv("TARGET_KL", "")
TARGET_KL = float(TARGET_KL_RAW) if TARGET_KL_RAW else None
CLIP_VLOSS = os.getenv("CLIP_VLOSS", "true").strip().lower() != "false"

DEFAULT_VEC_MODE = "sync" if os.name == "nt" else "async"
VEC_MODE = os.getenv("VEC_MODE", DEFAULT_VEC_MODE).strip().lower()

TARGET_ROLLOUT_SIZE = int(os.getenv("TARGET_ROLLOUT_SIZE", "16384"))
PPO_N_STEPS = int(
    os.getenv(
        "PPO_N_STEPS",
        str(max(128, TARGET_ROLLOUT_SIZE // max(N_WORLDS * CONTROLLED_VEHICLES, 1))),
    )
)


def make_writer(run_name: str):
    TB_DIR.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(TB_DIR / run_name))


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MONITOR_ROOT.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        run_name = mappo_run_name(
            SCENARIO_NAME,
            EXP_VERSION,
            CONTROLLED_VEHICLES,
            seed,
        )
        writer = make_writer(run_name)

        torch.manual_seed(seed)
        np.random.seed(seed)

        monitor_dir = MONITOR_ROOT / run_name
        monitor_dir.mkdir(parents=True, exist_ok=True)

        test_env = make_shared_world_env(
            scenario_name=SCENARIO_NAME,
            controlled_vehicles=CONTROLLED_VEHICLES,
        )
        test_obs, _ = test_env.reset(seed=seed)
        agent_obs = np.asarray(test_obs[0], dtype=np.float32)
        obs_dim = int(agent_obs.size)
        action_space = test_env.action_space.spaces[0]
        action_dim = int(np.prod(action_space.shape))
        state_dim = CONTROLLED_VEHICLES * obs_dim

        print(
            run_name,
            "agents:",
            len(test_obs),
            "agent obs shape:",
            agent_obs.shape,
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

        num_envs = env.num_envs
        batch_size = PPO_N_STEPS * num_envs
        minibatch_size = max(1, batch_size // max(NUM_MINIBATCHES, 1))
        num_updates = max(1, TOTAL_TIMESTEPS // batch_size)
        total_timesteps_actual = num_updates * batch_size

        print(
            f"{run_name}: worlds={N_WORLDS}, agents_per_world={CONTROLLED_VEHICLES}, "
            f"ppo_slots={num_envs}, vec_mode={VEC_MODE}, n_steps={PPO_N_STEPS}, "
            f"num_updates={num_updates}, total_timesteps_actual={total_timesteps_actual}, "
            f"reward_mode={REWARD_MODE}",
            flush=True,
        )

        config = MAPPOConfig(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=CONTROLLED_VEHICLES,
            hidden_dim=HIDDEN_DIM,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            update_epochs=UPDATE_EPOCHS,
            num_minibatches=NUM_MINIBATCHES,
            clip_vloss=CLIP_VLOSS,
            target_kl=TARGET_KL,
        )

        policy = MAPPOPolicy(
            config,
            action_low=np.asarray(action_space.low, dtype=np.float32),
            action_high=np.asarray(action_space.high, dtype=np.float32),
        ).to(DEVICE)
        optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=1e-5)

        obs = env.reset()
        global_step = 0

        obs_buf = np.zeros((PPO_N_STEPS, num_envs, obs_dim), dtype=np.float32)
        state_buf = np.zeros((PPO_N_STEPS, num_envs, state_dim), dtype=np.float32)
        action_buf = np.zeros((PPO_N_STEPS, num_envs, action_dim), dtype=np.float32)
        logprob_buf = np.zeros((PPO_N_STEPS, num_envs), dtype=np.float32)
        reward_buf = np.zeros((PPO_N_STEPS, num_envs), dtype=np.float32)
        done_buf = np.zeros((PPO_N_STEPS, num_envs), dtype=np.float32)
        value_buf = np.zeros((PPO_N_STEPS, num_envs), dtype=np.float32)
        agent_id_buf = np.zeros((PPO_N_STEPS, num_envs), dtype=np.int64)

        ep_returns = np.zeros((N_WORLDS,), dtype=np.float32)
        ep_lengths = np.zeros((N_WORLDS,), dtype=np.int32)
        completed_returns: deque[float] = deque(maxlen=100)
        completed_lengths: deque[int] = deque(maxlen=100)

        for update in range(1, num_updates + 1):
            for step in range(PPO_N_STEPS):
                local_obs, states, agent_ids = build_centralized_inputs(
                    obs,
                    num_agents=CONTROLLED_VEHICLES,
                )

                obs_buf[step] = local_obs
                state_buf[step] = states
                agent_id_buf[step] = agent_ids

                with torch.no_grad():
                    raw_actions_t, clipped_actions_t, log_probs_t, values_t = policy.act(
                        torch.as_tensor(local_obs, dtype=torch.float32, device=DEVICE),
                        torch.as_tensor(states, dtype=torch.float32, device=DEVICE),
                        torch.as_tensor(agent_ids, dtype=torch.long, device=DEVICE),
                        deterministic=False,
                    )

                raw_actions = raw_actions_t.cpu().numpy()
                clipped_actions = clipped_actions_t.cpu().numpy()
                log_probs = log_probs_t.cpu().numpy()
                values = values_t.cpu().numpy()

                action_buf[step] = raw_actions
                logprob_buf[step] = log_probs
                value_buf[step] = values

                next_obs, rewards, dones, infos = env.step(clipped_actions)
                slot_rewards = extract_slot_rewards(
                    rewards,
                    infos,
                    reward_mode=REWARD_MODE,
                )

                reward_buf[step] = slot_rewards
                done_buf[step] = dones.astype(np.float32)

                world_reward_step = world_metrics_from_slots(rewards, CONTROLLED_VEHICLES)
                world_done_step = world_metrics_from_slots(dones, CONTROLLED_VEHICLES)

                ep_returns += world_reward_step[:, 0]
                ep_lengths += 1

                for world_index, done_row in enumerate(world_done_step):
                    if not bool(done_row[0]):
                        continue
                    completed_returns.append(float(ep_returns[world_index]))
                    completed_lengths.append(int(ep_lengths[world_index]))
                    ep_returns[world_index] = 0.0
                    ep_lengths[world_index] = 0

                obs = next_obs
                global_step += num_envs

            with torch.no_grad():
                _, next_states, next_agent_ids = build_centralized_inputs(
                    obs,
                    num_agents=CONTROLLED_VEHICLES,
                )
                next_values = policy.critic(
                    torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(next_agent_ids, dtype=torch.long, device=DEVICE),
                ).cpu().numpy()

            advantages = np.zeros_like(reward_buf)
            lastgaelam = np.zeros((num_envs,), dtype=np.float32)
            for t in reversed(range(PPO_N_STEPS)):
                if t == PPO_N_STEPS - 1:
                    next_nonterminal = 1.0 - done_buf[t]
                    next_value = next_values
                else:
                    next_nonterminal = 1.0 - done_buf[t]
                    next_value = value_buf[t + 1]
                delta = reward_buf[t] + GAMMA * next_value * next_nonterminal - value_buf[t]
                lastgaelam = delta + GAMMA * GAE_LAMBDA * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + value_buf

            b_obs = torch.as_tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=DEVICE)
            b_states = torch.as_tensor(
                state_buf.reshape(-1, state_dim), dtype=torch.float32, device=DEVICE
            )
            b_actions = torch.as_tensor(
                action_buf.reshape(-1, action_dim), dtype=torch.float32, device=DEVICE
            )
            b_logprobs = torch.as_tensor(
                logprob_buf.reshape(-1), dtype=torch.float32, device=DEVICE
            )
            b_advantages = torch.as_tensor(
                advantages.reshape(-1), dtype=torch.float32, device=DEVICE
            )
            b_returns = torch.as_tensor(
                returns.reshape(-1), dtype=torch.float32, device=DEVICE
            )
            b_values = torch.as_tensor(
                value_buf.reshape(-1), dtype=torch.float32, device=DEVICE
            )
            b_agent_ids = torch.as_tensor(
                agent_id_buf.reshape(-1), dtype=torch.long, device=DEVICE
            )

            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std(unbiased=False) + 1e-8
            )

            clipfracs = []
            approx_kl = 0.0
            last_entropy_loss = 0.0
            last_policy_loss = 0.0
            last_value_loss = 0.0

            b_inds = np.arange(batch_size)
            for epoch in range(UPDATE_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    new_logprobs, entropies, new_values = policy.evaluate_actions(
                        b_obs[mb_inds],
                        b_states[mb_inds],
                        b_agent_ids[mb_inds],
                        b_actions[mb_inds],
                    )

                    logratio = new_logprobs - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = float(((ratio - 1.0) - logratio).mean().cpu().item())
                        clipfracs.append(
                            float(
                                ((ratio - 1.0).abs() > CLIP_RANGE)
                                .float()
                                .mean()
                                .cpu()
                                .item()
                            )
                        )

                    mb_advantages = b_advantages[mb_inds]
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - CLIP_RANGE,
                        1.0 + CLIP_RANGE,
                    )
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if CLIP_VLOSS:
                        value_pred_clipped = b_values[mb_inds] + (
                            new_values - b_values[mb_inds]
                        ).clamp(-CLIP_RANGE, CLIP_RANGE)
                        value_losses = (new_values - b_returns[mb_inds]) ** 2
                        value_losses_clipped = (
                            value_pred_clipped - b_returns[mb_inds]
                        ) ** 2
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * ((new_values - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropies.mean()
                    loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                    last_entropy_loss = float(entropy_loss.detach().cpu().item())
                    last_policy_loss = float(policy_loss.detach().cpu().item())
                    last_value_loss = float(value_loss.detach().cpu().item())

                if TARGET_KL is not None and approx_kl > TARGET_KL:
                    break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            ev = explained_variance(y_pred, y_true)

            mean_ep_rew = float(np.mean(completed_returns)) if completed_returns else float("nan")
            mean_ep_len = float(np.mean(completed_lengths)) if completed_lengths else float("nan")

            writer.add_scalar("charts/learning_rate", LEARNING_RATE, global_step)
            writer.add_scalar("charts/global_step", global_step, global_step)
            writer.add_scalar("losses/policy_loss", last_policy_loss, global_step)
            writer.add_scalar("losses/value_loss", last_value_loss, global_step)
            writer.add_scalar("losses/entropy", last_entropy_loss, global_step)
            writer.add_scalar("losses/approx_kl", approx_kl, global_step)
            writer.add_scalar(
                "losses/clipfrac",
                float(np.mean(clipfracs)) if clipfracs else 0.0,
                global_step,
            )
            if not np.isnan(ev):
                writer.add_scalar("losses/explained_variance", ev, global_step)
            if not np.isnan(mean_ep_rew):
                writer.add_scalar("rollout/ep_rew_mean", mean_ep_rew, global_step)
            if not np.isnan(mean_ep_len):
                writer.add_scalar("rollout/ep_len_mean", mean_ep_len, global_step)

            print(
                f"update={update}/{num_updates} "
                f"global_step={global_step} "
                f"ep_rew_mean={mean_ep_rew:.3f} "
                f"ep_len_mean={mean_ep_len:.3f} "
                f"policy_loss={last_policy_loss:.4f} "
                f"value_loss={last_value_loss:.4f} "
                f"entropy={last_entropy_loss:.4f} "
                f"approx_kl={approx_kl:.5f} "
                f"clipfrac={(float(np.mean(clipfracs)) if clipfracs else 0.0):.4f} "
                f"explained_variance={ev:.4f}",
                flush=True,
            )

        metadata = {
            "run_name": run_name,
            "scenario_name": SCENARIO_NAME,
            "exp_version": EXP_VERSION,
            "controlled_vehicles": CONTROLLED_VEHICLES,
            "seed": seed,
            "reward_mode": REWARD_MODE,
            "num_worlds": N_WORLDS,
            "ppo_n_steps": PPO_N_STEPS,
            "action_low": np.asarray(action_space.low, dtype=np.float32).tolist(),
            "action_high": np.asarray(action_space.high, dtype=np.float32).tolist(),
        }
        save_mappo_checkpoint(
            MODEL_DIR / f"{run_name}.pt",
            policy=policy,
            optimizer=optimizer,
            metadata=metadata,
        )

        env.close()
        writer.close()
