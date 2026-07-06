import sys
from pathlib import Path

# Add the root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gymnasium as gym
import torch

import custom_slimevolleygym
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from imitation.util.util import make_vec_env
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


rng = np.random.default_rng(0)
env = make_vec_env(
    "SlimeVolley-v0",
    rng=rng,
    n_envs=4,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)], 
)

class SlimeVolleyExpertPolicy(BasePolicy):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super().__init__(observation_space, action_space)
        # Store a separate expert model for each environment
        self.expert_models = None
        self.n_envs = 1

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        # Convert torch tensor to numpy for the expert policy
        obs_np = observation.cpu().numpy() if isinstance(observation, torch.Tensor) else observation

        if obs_np.ndim > 1:
            # Vectorized environment case - each environment needs its own expert state
            n_envs = obs_np.shape[0]
            
            # Initialize expert models on first call
            if self.expert_models is None or len(self.expert_models) != n_envs:
                self.expert_models = [custom_slimevolleygym.BaselinePolicy() for _ in range(n_envs)]
                self.n_envs = n_envs
            
            actions = np.array([self.expert_models[i].predict(obs_np[i]) for i in range(n_envs)], dtype=np.float32)
        else:
            # Single environment case
            if self.expert_models is None:
                self.expert_models = [custom_slimevolleygym.BaselinePolicy()]
            actions = np.array(self.expert_models[0].predict(obs_np), dtype=np.float32)
        
        return torch.as_tensor(actions, dtype=torch.float32)
    

expert = PPO.load("ppo/best_model.zip", env=env)
reward, _ = evaluate_policy(expert, env, 10)
print("Reward:", reward)


print("Generating initial rollouts...")
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=100),
    rng=rng,
)

print("Training initial BC policy...")
transitions = rollout.flatten_trajectories(rollouts)

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
)

learner_rewards_before_training, _ = evaluate_policy(learner, env, 20, return_episode_rewards=True)


print("Training BC on initial demonstrations...")
gail_trainer.train(200_000)

print("\n=== Evaluation ===")
learner_rewards_after_training, _ = evaluate_policy(learner, env, 20, return_episode_rewards=True)

print(
    "Rewards before training:",
    np.mean(learner_rewards_before_training),
    "+/-",
    np.std(learner_rewards_before_training),
)

print(
    "Rewards after training:",
    np.mean(learner_rewards_after_training),
    "+/-",
    np.std(learner_rewards_after_training),
)

env = gym.make("SlimeVolley-v0", render_mode="human")
model = gail_trainer.gen_algo.policy
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
env.close()

# Save learned policy
model_path = "src/imitation_learning/imitation_models/gail_slimevolley_policy.zip"
gail_trainer.gen_algo.save(model_path)
print(f"Saved GAIL policy to {model_path}")