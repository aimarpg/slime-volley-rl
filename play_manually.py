import gymnasium as gym

import custom_slimevolleygym
import numpy as np
import pygame

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from imitation.algorithms.bc import BC

env = gym.make("SlimeVolley-v0", render_mode="human")

pygame.init()
model = PPO.load("ppo/final_model")
obs, info = env.reset()
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    manual_action = [0.0, 0.0, 0.0]  # [left, right, jump]
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        manual_action[0] = 1
    if keys[pygame.K_RIGHT]:
        manual_action[1] = 1
    if keys[pygame.K_UP]:
        manual_action[2] = 1

    obs, reward, terminated, truncated, info = env.step(manual_action)
    env.render()
    done = terminated or truncated

env.close()
