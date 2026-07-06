import gymnasium as gym

import custom_slimevolleygym
import numpy as np

from stable_baselines3 import PPO


env = gym.make("SlimeVolley-v0", render_mode="human")

model = PPO.load("ppo/best_model.zip", env=env)

class BaselinePolicyWrapper(custom_slimevolleygym.BaselinePolicy):
    def predict(self, observation):
        return super().predict(observation), None

#model = BaselinePolicyWrapper()
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

env.close()
