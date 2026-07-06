import os
import gymnasium as gym

import custom_slimevolleygym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


from shutil import copyfile 

# Settings
SEED = 17
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e5)
EVAL_EPISODES = 100
BEST_THRESHOLD = 0.5 # must achieve a mean score above this to replace prev best self

LOGDIR = "ppo_selfplay"

class SlimeVolleySelfPlayEnv(custom_slimevolleygym.SlimeVolleyEnv):
  def __init__(self):
    super(SlimeVolleySelfPlayEnv, self).__init__()
    self.policy = self
    self.best_model = None
    self.best_model_filename = None
    
  def predict(self, obs): 
    if self.best_model is None:
      return self.action_space.sample()
    else:
      action, _ = self.best_model.predict(obs)
      return action
    
  def reset(self, seed=None, options=None, **kwargs):
    modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
    modellist.sort()
    if len(modellist) > 0:
      filename = os.path.join(LOGDIR, modellist[-1]) # the latest best model
      if filename != self.best_model_filename:
        print("loading model: ", filename)
        self.best_model_filename = filename
        if self.best_model is not None:
          del self.best_model
        self.best_model = PPO.load(filename, env=self)
    return super(SlimeVolleySelfPlayEnv, self).reset(seed=seed, options=options, **kwargs)

class SelfPlayCallback(EvalCallback):
  # Only save new version of best model if beats prev self by BEST_THRESHOLD score
  # after saving model, resets the best score to be BEST_THRESHOLD
  def __init__(self, *args, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    self.best_mean_reward = BEST_THRESHOLD
    self.generation = 0
    
  def _on_step(self) -> bool:    
    result = super(SelfPlayCallback, self)._on_step()

    if result and self.best_mean_reward > BEST_THRESHOLD:
      self.generation += 1
      print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
      print("SELFPLAY: new best model, bumping up generation to", self.generation)
      source_file = os.path.join(LOGDIR, "best_model.zip")
      backup_file = os.path.join(LOGDIR, "history_"+str(self.generation).zfill(8)+".zip")
      copyfile(source_file, backup_file)
      self.best_mean_reward = BEST_THRESHOLD
    return result

def rollout(env, policy):
  obs = env.reset()

  done = False
  total_reward = 0

  while not done:

    action, _ = policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    total_reward += reward

  return total_reward

def train():
  os.makedirs(LOGDIR, exist_ok=True)

  env = custom_slimevolleygym.SurvivalRewardEnv(SlimeVolleySelfPlayEnv())
  env.reset(seed=SEED)

  model = PPO(MlpPolicy, env, n_steps=4096, n_epochs=10, learning_rate=3e-4, batch_size=64, verbose=2, tensorboard_log=LOGDIR)

  eval_callback = SelfPlayCallback(env,
    best_model_save_path=LOGDIR,
    log_path=LOGDIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=False)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  model.save(os.path.join(LOGDIR, "final_model")) 

  env.close()

if __name__=="__main__":
  train()