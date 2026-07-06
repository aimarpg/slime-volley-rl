import gymnasium as gym
import numpy as np

import custom_slimevolleygym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

rng = np.random.default_rng(0)
env = DummyVecEnv(
    [lambda: gym.make("SlimeVolley-v0")] * 30
)

expert = PPO.load("best_model.zip", env=env)
reward, _ = evaluate_policy(expert, env, 30)
print("Reward:", reward)