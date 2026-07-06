import os

import custom_slimevolleygym
from custom_slimevolleygym import SlimeVolleyEnv

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

NUM_TIMESTEPS = int(2e7)
SEED = 0
EVAL_FREQ = 250000
EVAL_EPISODES = 100
LOGDIR = "ppo"

logger.configure(folder=LOGDIR)

env = make_vec_env("SlimeVolley-v0", seed=SEED)

model = PPO(MlpPolicy, env, clip_range=0.2, learning_rate=3e-4, batch_size=64, 
            gamma=0.99, gae_lambda=0.95, verbose=2, tensorboard_log=LOGDIR)

eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

model.save(os.path.join(LOGDIR, "final_model")) 

env.close()