import os
import numpy as np
import torch
import ray
import pandas as pd

from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from pettingzoo.utils.conversions import parallel_to_aec
from ray.rllib.env import PettingZooEnv

from envs.dynamic_ultimate_frisbee import DynamicUltimateFrisbeeEnv


def rllib_env_creator(env_config=None):
    base_env = DynamicUltimateFrisbeeEnv(
        num_players_per_team=7,
        use_regulation_field=True,
        seed=0,
        max_steps=800,
        debug=False,
    )
    return PettingZooEnv(parallel_to_aec(base_env))


register_env("frisbee_env", rllib_env_creator)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, local_mode=False)

    checkpoint_path = "/Users/adam-admin/code/Multi-Agent-Frisbee/checkpoints/best"

    algo = PPO.from_checkpoint(checkpoint_path)
    module = algo.get_module("shared_policy")

    env = DynamicUltimateFrisbeeEnv(
        num_players_per_team=7,
        use_regulation_field=True,
        seed=123,
        max_steps=800,
        debug=True,
    )

    num_episodes = 15
    max_steps = 800

    episode_log = []

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        episode_seed = np.random.randint(0, 10_000_000)

        obs, _ = env.reset(seed=episode_seed)
        terminated = {a: False for a in env.agents}
        truncated = {a: False for a in env.agents}

        for step in range(max_steps):
            actions = {}

            for agent, agent_obs in obs.items():
                obs_arr = np.asarray(agent_obs, dtype=np.float32)
                batch = {"obs": torch.tensor(obs_arr).unsqueeze(0)}

                out = module.forward_inference(batch)
                raw = out["action_dist_inputs"][0].detach().cpu().numpy()

                half = raw.shape[0] // 2
                ax, ay, throw_raw = raw[:half]

                actions[agent] = np.array([ax, ay, throw_raw], dtype=np.float32)

            obs, rewards, terminated, truncated, _ = env.step(actions)
            env.render_matplotlib(block=False)

            if all(terminated.values()) or all(truncated.values()):
                print(f"Episode ended at step {step + 1}")
                break

        episode_log.append({
            "score": env.episode_score,
            "throws": env.episode_throw_count,
            "catches": env.episode_catch_count,
            "intercepts": env.episode_intercept_count,
            "stallouts": env.episode_stallouts,
            "reason": env.episode_reason,
        })

    env.close()
    ray.shutdown()

    df = pd.DataFrame(episode_log)
    df.to_csv("/Users/adam-admin/code/Multi-Agent-Frisbee/eval_episode_stats.csv", index=False)

    print("Saved evaluation stats to eval_episode_stats.csv")
