import sys
from pathlib import Path

# Add the root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tempfile
import gymnasium as gym
import torch

import custom_slimevolleygym
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy

from imitation.util.util import make_vec_env
from imitation.data import rollout
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer, LinearBetaSchedule
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.logger import configure
from imitation.policies.serialize import load_policy

rng = np.random.default_rng(0)
env = make_vec_env(
    "SlimeVolley-v0",
    rng=rng,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)], 
)

expert = load_policy("ppo", env, path="ppo/best_model.zip")
reward, _ = evaluate_policy(expert, env, 10)
print("Reward:", reward)

# Configure TensorBoard logging
log_dir = "src/imitation_learning/imitation_logs/dagger"
bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
    batch_size=128,
    custom_logger=configure(log_dir, ["stdout", "tensorboard"])
)

with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng,
        beta_schedule=LinearBetaSchedule(8),
    )
    dagger_trainer.train(40_000)

print("\n=== Evaluation ===")
reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
print(f"BC Reward: {reward}")
