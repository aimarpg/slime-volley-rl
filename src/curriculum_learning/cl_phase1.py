import sys
import os
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../..")))

import gym
import math
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from custom_slimevolleygym import slimevolley0v1 as slimevolley
from cl_helpers import RenderCallback, watch, setup_directories, DiscreteWrapper

# ==========================================
# 1. PHASE 1 WRAPPER (static opponent)
# ==========================================
class Phase1Wrapper(gym.Wrapper):
    """
    Wrapper for Phase 1:
    - Opponent: STATIC.
    - Aids: Delta-Distance and Touch Bonus to teach movement.
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_dist = 0.0

    def reset(self):
        obs = self.env.reset()
        game = self.env.unwrapped.game
        self.prev_dist = self._get_distance(game)
        return obs

    def step(self, action):
        # --- PHASE 1: STATIC OPPONENT ---
        static_action = [0, 0, 0]
        obs, reward, done, info = self.env.step(action, otherAction=static_action)
        
        game = self.env.unwrapped.game
        curr_dist = self._get_distance(game)
        
        # 1. Delta Reward (Move towards ball)
        delta = self.prev_dist - curr_dist
        reward += delta * 0.05 
        
        # 2. Touch Bonus
        sum_radii = game.ball.r + game.agent_right.r
        if curr_dist < sum_radii + 0.1:
            reward += 0.015
        
        self.prev_dist = curr_dist
        return obs, reward, done, info

    def _get_distance(self, game):
        dx = game.ball.x - game.agent_right.x
        dy = game.ball.y - game.agent_right.y
        return math.sqrt(dx*dx + dy*dy)

# ==========================================
# 2. TRAINING FUNCTION
# ==========================================
def train_phase1(total_timesteps=200000, n_envs=4, render=False, experiment_name="phase1", algo="PPO"):
    
    final_exp_name = f"{experiment_name}_{algo.lower()}"
    log_dir, models_dir = setup_directories(experiment_name, "CL_Phase1", algo)

    print(f"--- Starting PHASE 1: Static Opponent ({algo}) ---")
    print(f"--- TensorBoard: {log_dir}")
    print(f"--- Models: {models_dir}")

    # Configure Algorithm
    if algo == "PPO":
        AlgoClass = PPO
        algo_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 4096 // n_envs,
            "batch_size": 64 * n_envs
        }
    elif algo == "DQN":
        AlgoClass = DQN
        n_envs = 1
        algo_kwargs = {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05
        }

    if render:
        print("Render enabled: Forcing n_envs=1.")
        n_envs = 1
    
    def make_env():
        env = slimevolley0v1.SlimeVolleyEnv()
        # 1. Apply Logic Wrapper (Handles Game Physics/Rewards)
        env = Phase1Wrapper(env)
        
        # 2. Apply Discrete Adapter (Only for DQN)
        # This must be the OUTER wrapper so the Agent sees Discrete actions
        if algo == "DQN":
            env = DiscreteWrapper(env)
        return env
    
    # Vectorized environment
    env = make_vec_env(make_env, n_envs=n_envs)
    
    # Evaluation environment
    eval_env = make_env()

    eval_freq = 2000 if total_timesteps < 20000 else 10000

    # EVALUATION CALLBACK
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Initialize Model using the selected Class
    model = AlgoClass("MlpPolicy", env, verbose=1, 
                      tensorboard_log=log_dir,
                      **algo_kwargs)

    callbacks = [eval_callback]
    if render:
        callbacks.append(RenderCallback(True))

    try:
        model.learn(total_timesteps=total_timesteps, 
                    callback=callbacks,
                    tb_log_name=final_exp_name,
                    reset_num_timesteps=True)
        
        print("Training finished.")
        
        final_path = os.path.join(models_dir, "final_model")
        model.save(final_path)
        print(f"FINAL Model: {final_path}.zip")
        
    except KeyboardInterrupt:
        print("\nInterrupted.")
        int_path = os.path.join(models_dir, "interrupted_model")
        model.save(int_path)
    finally:
        env.close()
        eval_env.close()

# ==========================================
# 3. MAIN BLOCK
# ==========================================
if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    MODE = "WATCH" 
    EXPERIMENT_NAME = "phase1" 
    ALGO = "DQN"
    
    STEPS = 2000000
    CORES = 4
    # ---------------------

    if MODE == "TRAIN":
        train_phase1(total_timesteps=STEPS, 
                     n_envs=CORES, 
                     render=False, 
                     experiment_name=EXPERIMENT_NAME,
                     algo=ALGO) 
        
    elif MODE == "WATCH":
        path = os.path.join("./model_logs", f"{ALGO.lower()}/{EXPERIMENT_NAME}", "best_model")
        watch(path, wrapper_class=Phase1Wrapper, algo=ALGO)