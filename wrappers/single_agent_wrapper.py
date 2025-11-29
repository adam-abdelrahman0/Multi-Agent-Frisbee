# wrappers/single_agent_wrapper.py
import gymnasium as gym
import numpy as np
from pettingzoo.utils.wrappers import BaseWrapper

class SingleAgentOffenseWrapper(BaseWrapper, gym.Env):
    """
    Wrap a PettingZoo multi-agent environment so that a single offense agent
    (team_0) can be trained with SB3.
    """

    def __init__(self, env, agent_id="team_0_player_0"):
        super().__init__(env)
        self.agent_id = agent_id

        # Force a fixed observation shape based on current env
        obs_example = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_example.shape, dtype=np.float32
        )
        # Discrete(5): up, down, left, right, throw
        self.action_space = gym.spaces.Discrete(5)

    def _get_obs(self):
        """Always returns fixed-length observation for the selected agent."""
        all_obs = self.env._get_obs()
        obs = all_obs[self.agent_id]
        return np.array(obs, dtype=np.float32)

    def reset(self, **kwargs):
        """Reset underlying environment and return obs, info for SB3."""
        observations, infos = self.env.reset(**kwargs)
        obs = np.array(observations[self.agent_id], dtype=np.float32)
        info = infos.get(self.agent_id, {})
        return obs, info

    def step(self, action):
        """Step environment and return Gym-style outputs."""
        actions = {self.agent_id: action}
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        obs = np.array(observations[self.agent_id], dtype=np.float32)
        reward = rewards[self.agent_id]
        terminated = terminations[self.agent_id]
        truncated = truncations[self.agent_id]
        info = infos.get(self.agent_id, {})

        return obs, reward, terminated, truncated, info

