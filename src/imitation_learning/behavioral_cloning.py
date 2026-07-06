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

from imitation.util.util import make_vec_env
from imitation.data import rollout
from imitation.algorithms.bc import BC
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.logger import configure

rng = np.random.default_rng(0)
env = make_vec_env(
    "SlimeVolley-v0",
    rng=rng,
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
            # Vectorized environment, each environment needs its own expert state
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
    

expert = SlimeVolleyExpertPolicy(env.observation_space, env.action_space)
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

print("Number of transitions:", len(transitions))

# Configure TensorBoard logging
log_dir = "src/imitation_learning/imitation_logs/bc"
bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
    batch_size=128,
    custom_logger=configure(log_dir, ["stdout", "tensorboard"]),
)

print("Training BC on initial demonstrations...")
bc_trainer.train(n_epochs=5)

print("\n=== Evaluation ===")
reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"BC Reward: {reward}")

# Save learned policy
model_path = "src/imitation_learning/imitation_models/bc_slimevolley_policy.zip"
bc_trainer.policy.save(model_path)
print(f"Saved BC policy to {model_path}")

""" env = gym.make("SlimeVolley-v0", render_mode="human")
model = bc_trainer.policy
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
env.close()
 """