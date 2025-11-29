import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from pettingzoo.utils.conversions import parallel_to_aec
from ray.rllib.env import PettingZooEnv

from envs.dynamic_ultimate_frisbee import DynamicUltimateFrisbeeEnv


def env_creator(env_config=None):
    base_env = DynamicUltimateFrisbeeEnv(
        num_players_per_team=5,      # 5 v 5
        use_regulation_field=True,   # 110 × 40 field
        seed=0,
        max_steps=800,               # longer episode for real field
    )
    return PettingZooEnv(parallel_to_aec(base_env))


register_env("frisbee_env", env_creator)

temp = env_creator()
example_agent = temp.possible_agents[0]

obs_space = temp.observation_space[example_agent]
act_space = temp.action_space[example_agent]

temp.close()



if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment(env="frisbee_env")
        .framework("torch")
        .env_runners(num_env_runners=2)
        .training(
            gamma=0.99,
            lr=1e-3,
            train_batch_size=8128,
            minibatch_size=1024,
            num_epochs=5,
            vf_clip_param=10.0,
            model={"vf_share_layers": True},
        )
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    policy_class=None,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={}
                )
            },
            policy_mapping_fn=lambda *_: "shared_policy",
            policies_to_train=["shared_policy"],
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    for i in range(300):
        result = algo.train()
        mean_reward = (
            result.get("env_runners", {}).get("episode_return_mean")
            or result.get("episode_return_mean")
            or 0.0
        )
        print(f"Iter {i} | mean reward: {mean_reward:.3f}")

        if i % 50 == 0:
            ckpt = algo.save("/Users/adam-admin/code/Multi-Agent-Frisbee/checkpoints")
            print(f"Checkpoint saved: {ckpt}")

    ray.shutdown()
