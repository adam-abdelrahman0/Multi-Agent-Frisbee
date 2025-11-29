import os
import numpy as np
import torch
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from pettingzoo.utils.conversions import parallel_to_aec
from ray.rllib.env import PettingZooEnv

from envs.dynamic_ultimate_frisbee import DynamicUltimateFrisbeeEnv


def env_creator(env_config=None):
    base_env = DynamicUltimateFrisbeeEnv(
        num_players_per_team=5,
        use_regulation_field=True,
        seed=0,
        max_steps=800,
    )
    return PettingZooEnv(parallel_to_aec(base_env))


register_env("frisbee_env", env_creator)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, local_mode=True)

    checkpoint_path = "/Users/adam-admin/code/Multi-Agent-Frisbee/checkpoints"
    algo = PPO.from_checkpoint(checkpoint_path)

    module = algo.get_module("shared_policy")

    env = env_creator()
    obs, _ = env.reset()

    print("\nENV STRUCTURE:")
    cur = env
    for i in range(10):
        print(f"Layer {i}: {type(cur)}")
        cur = getattr(cur, "env", None) or getattr(cur, "unwrapped", None)
        if cur is None:
            break

    print("\nSearching for render_matplotlib ...")
    cur = env
    for i in range(10):
        if hasattr(cur, "render_matplotlib"):
            print(f"FOUND on layer {i}: {type(cur)}")
            ENV_RENDER_OBJ = cur
            break
        cur = getattr(cur, "env", None) or getattr(cur, "unwrapped", None)



    for _ in range(500):
        actions = {}
        for agent, agent_obs in obs.items():
            arr = np.asarray(agent_obs, dtype=np.float32)

            batch = {"obs": torch.tensor(arr).unsqueeze(0)}
            out = module.forward_inference(batch)

            logits = out["action_dist_inputs"]
            action = int(torch.argmax(logits, dim=-1).item())

            actions[agent] = action

        obs, rewards, terminated, truncated, _ = env.step(actions)

        env.env.env.env.render_matplotlib(block=False)

        if all(terminated.values()) or all(truncated.values()):
            break

    env.env.env.env.render_matplotlib(block=True)

    env.close()
    ray.shutdown()
