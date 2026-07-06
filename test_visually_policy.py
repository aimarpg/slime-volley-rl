import gymnasium as gym
import torch

import custom_slimevolleygym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from imitation.algorithms.bc import BC
from stable_baselines3.common.policies import BasePolicy

env = gym.make("SlimeVolley-v0", render_mode="human")
    
class SlimeVolleyExpertPolicy(BasePolicy):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super().__init__(observation_space, action_space)
        self.expert_model = custom_slimevolleygym.BaselinePolicy()

    def _predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # Handle vectorized environments by processing each observation individually
        if observation.ndim > 1:
            actions = torch.asarray([self.expert_model.predict(obs) for obs in observation])
        else:
            actions = self.expert_model.predict(observation)
        return actions
    

model = SlimeVolleyExpertPolicy(env.observation_space, env.action_space)
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

env.close()
