import sys
import os
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../..")))

import gym
import math
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from custom_slimevolleygym import slimevolley0v1 as slimevolley
from cl_helpers import RenderCallback, watch, setup_directories, DiscreteWrapper

# ==============================================
# 1. PHASE 2 WRAPPER (basic heuristic opponent)
# ==============================================
class Phase2Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_dist = 0.0

    def reset(self):
        obs = self.env.reset()
        game = self.env.unwrapped.game
        self._hack_serve(game)
        self.prev_dist = self._get_distance(game)
        return obs

    def step(self, action):
        game = self.env.unwrapped.game
        ball = game.ball
        opp = game.agent_left

        # --- BASIC BOT LOGIC ---
        bot_action = [0, 0, 0]

        # A. Horizontal Movement
        if np.random.rand() > 0.0: 
            if ball.x > opp.x: 
                bot_action = [1, 0, 0] # Right
            elif ball.x < opp.x:
                bot_action = [0, 1, 0] # Left
        
        # B. Jump (Adjusted)
        dist_sq = (ball.x - opp.x)**2 + (ball.y - opp.y)**2
        if dist_sq < 8.0 and ball.y > 2:
             if np.random.rand() > 0.0: 
                 bot_action[2] = 1

        # Execute step
        obs, reward, done, info = self.env.step(action, otherAction=bot_action)
        
        # --- SERVE HACK ---
        if reward != 0 and not done:
            self._hack_serve(game)

        # Rewards
        curr_dist = self._get_distance(game)
        delta = self.prev_dist - curr_dist
        reward += delta * 0.05 
        
        sum_radii = game.ball.r + game.agent_right.r
        if curr_dist < sum_radii + 0.1:
            reward += 0.015
        
        self.prev_dist = curr_dist
        return obs, reward, done, info

    def _hack_serve(self, game):
        ball = game.ball
        ball.x = 0.0
        ball.y = 8.0
        ball.vx = np.random.uniform(12.0, 40.0)
        ball.vy = np.random.uniform(10.0, 60.0)

    def _get_distance(self, game):
        dx = game.ball.x - game.agent_right.x
        dy = game.ball.y - game.agent_right.y
        return math.sqrt(dx*dx + dy*dy)

# ==========================================
# 2. TRAINING FUNCTION
# ==========================================
def train_phase2(total_timesteps=1000000, n_envs=4, render=False, 
                 experiment_name="phase2", previous_model="phase1", algo="PPO"):
    
    # Setup directories (ahora incluye el algo en el nombre)
    final_exp_name = f"{experiment_name}_{algo.lower()}"
    log_dir, models_dir = setup_directories(experiment_name, "CL_Phase2", algo)

    print(f"--- Starting PHASE 2: Forced Serve + Basic Bot ({algo}) ---")
    print(f"--- TensorBoard: {log_dir}")
    print(f"--- Models: {models_dir}")

    # Configure Algorithm Class
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
            "exploration_initial_eps": 0.1,
            "exploration_final_eps": 0.05
        }

    if render:
        print("Render enabled: Forcing n_envs=1.")
        n_envs = 1
    
    def make_env():
        env = slimevolley.SlimeVolleyEnv()
        # 1. Logic Wrapper
        env = Phase2Wrapper(env)
        # 2. Discrete Adapter (Only for DQN)
        if algo == "DQN":
            env = DiscreteWrapper(env)
        return env
    
    # Vectorized environment
    env = make_vec_env(make_env, n_envs=n_envs)
    eval_env = make_env()

    # Frequency adjustment
    eval_freq = 2000 if total_timesteps < 20000 else 10000

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # ==========================================
    # LOAD PREVIOUS MODEL (PHASE 1)
    # ==========================================
    # Path: ./model_logs/dqn/phase1/best_model.zip
    prev_exp_full_name = f"{algo.lower()}/{previous_model}"
    prev_model_path = os.path.join("./model_logs", prev_exp_full_name, "best_model.zip")
    
    if not os.path.exists(prev_model_path):
        # Intentar con final_model
        alt_path = os.path.join("./model_logs", prev_exp_full_name, "final_model.zip")
        if os.path.exists(alt_path):
            prev_model_path = alt_path
        else:
            print(f"CRITICAL ERROR: Phase 1 model not found.")
            print(f"   Expected at: {prev_model_path}")
            print(f"   Or at:       {alt_path}")
            return

    print(f"Loading previous model from: {prev_model_path}")

    # Cargar el modelo pasando el entorno nuevo
    # Nota: custom_objects puede ser necesario si cambias versiones de python, pero suele ir bien directo
    model = AlgoClass.load(prev_model_path, env=env, tensorboard_log=log_dir, **algo_kwargs)

    callbacks = [eval_callback]
    if render:
        callbacks.append(RenderCallback(True))

    try:
        model.learn(total_timesteps=total_timesteps, 
                    callback=callbacks,
                    tb_log_name=final_exp_name,
                    reset_num_timesteps=True) # Reset timesteps para ver graficas limpias desde 0 en esta fase
        
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
    
    CURRENT_EXP_NAME = "phase2"     
    PREV_MODEL_NAME = "phase1"
    
    ALGO = "DQN"  # O "PPO"

    STEPS = 2000000 
    CORES = 4
    # ---------------------

    if MODE == "TRAIN":
        train_phase2(total_timesteps=STEPS,
                     n_envs=CORES, 
                     render=False, 
                     experiment_name=CURRENT_EXP_NAME,
                     previous_model=PREV_MODEL_NAME,
                     algo=ALGO)
        
    elif MODE == "WATCH":
        path = os.path.join("./model_logs", f"{ALGO.lower()}/{CURRENT_EXP_NAME}", "best_model")
        watch(path, wrapper_class=Phase2Wrapper, algo=ALGO)