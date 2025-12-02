import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from pettingzoo.utils.conversions import parallel_to_aec
from ray.rllib.env import PettingZooEnv

from envs.dynamic_ultimate_frisbee import DynamicUltimateFrisbeeEnv


def env_creator(env_config=None):
    base_env = DynamicUltimateFrisbeeEnv(
        num_players_per_team=4,
        use_regulation_field=True,
        seed=0,
        max_steps=800,
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
        .env_runners(num_env_runners=12)
        .training(
            gamma=0.995,
            lr=3e-4,
            entropy_coeff=0.01,
            train_batch_size=32768,
            minibatch_size=4096,
            num_epochs=10,
            grad_clip=1.0,
            model={
                "fcnet_hiddens": [512, 512],
                "vf_share_layers": True,
            },
        )
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
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

    best_reward = float("-inf")

    for i in range(50):
        result = algo.train()

        mean_reward = (
            result.get("env_runners", {}).get("episode_return_mean")
            or result.get("episode_return_mean")
            or 0.0
        )

        print(f"Iter {i} | mean reward: {mean_reward:.3f}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            ckpt = algo.save("/Users/adam-admin/code/Multi-Agent-Frisbee/checkpoints/best")
            print(f"New best checkpoint saved at iter {i} | reward {best_reward:.3f}")

        if i % 50 == 0:
            ckpt = algo.save(f"/Users/adam-admin/code/Multi-Agent-Frisbee/checkpoints/iter_{i}")
            print(f"Checkpoint saved: {ckpt}")

    ray.shutdown()
