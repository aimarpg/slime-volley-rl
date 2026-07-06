import sys
import os
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../..")))

import gym
import glob
import shutil
import time
import numpy as np
import random
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from custom_slimevolleygym import slimevolley0v1 as slimevolley
from custom_slimevolleygym.slimevolley0v1 import BaselinePolicy
from cl_helpers import RenderCallback, setup_directories, DiscreteWrapper

# ==============================================================================
# 1. PHASE 4 WRAPPER (baseline warm-up + iterative self-play league)
# ==============================================================================
class Phase4Wrapper(gym.Wrapper):
    """
    Wrapper for League training.
    Manages opponents: Baseline (Warmup) + History (League).
    Universal: Can load PPO or DQN opponents.
    """
    def __init__(self, env, history_dir, AlgoClass):
        super().__init__(env)
        self.history_dir = history_dir
        self.AlgoClass = AlgoClass
        self.baseline_policy = BaselinePolicy()
        self.opponent_pool = []
        self.current_opponent = None
        self.league_unlocked = False 
        self.refresh_pool()

    def refresh_pool(self):
        if os.path.exists(self.history_dir):
            self.opponent_pool = glob.glob(os.path.join(self.history_dir, "*.zip"))
            self.opponent_pool.sort()

    def add_new_opponent(self, path):
        if path not in self.opponent_pool:
            self.opponent_pool.append(path)

    def activate_league(self):
        self.league_unlocked = True

    def reset(self):
        if not self.league_unlocked:
            # Warmup period: only baseline
            self.current_opponent = self.baseline_policy
        else:
            # League period
            # 30% baseline, otherwise random historical opponent
            if random.random() < 0.30:
                self.current_opponent = self.baseline_policy
            else:
                if len(self.opponent_pool) > 0:
                    opp_path = random.choice(self.opponent_pool)
                    try:
                        self.current_opponent = self.AlgoClass.load(opp_path, device="cpu")
                    except:
                        self.current_opponent = self.baseline_policy
        return self.env.reset()

    def step(self, action):
        opponent_action = None # Por defecto None (activa el bot interno del juego)

        # Solo calculamos acción manual si NO es la baseline
        if self.current_opponent is not None and self.current_opponent != self.baseline_policy:
            game = self.env.unwrapped.game
            opp_obs = self._get_opponent_observation(game)
            
            try:
                # Los modelos cargados (PPO/DQN) sí usan predict
                raw_action, _ = self.current_opponent.predict(opp_obs, deterministic=True)
                
                # Gestión de escalares (DQN) vs Vectores (PPO)
                if np.isscalar(raw_action) or np.ndim(raw_action) == 0:
                    opponent_action = DiscreteWrapper.to_vector(raw_action)
                else:
                    opponent_action = raw_action
            except Exception as e:
                # --- AVISO POR CONSOLA AÑADIDO ---
                print(f"======================================================================================")
                print(f"⚠️ [LEAGUE EXCEPTION] El oponente falló al predecir. Error: {e}")
                print(f"   -> Forzando acción [0,0,0] (AFK)")
                # ---------------------------------
                opponent_action = [0, 0, 0]

        # Si opponent_action es None, el env usa la Baseline Policy automáticamente
        obs, reward, done, info = self.env.unwrapped.step(action, otherAction=opponent_action)
        
        return obs, reward, done, info

    def _get_opponent_observation(self, game):
        obs = np.zeros(12)
        
        obs[0] = -game.agent_left.x; 
        obs[1] = game.agent_left.y
        obs[2] = -game.agent_left.vx; 
        obs[3] = game.agent_left.vy

        obs[4] = -game.ball.x; 
        obs[5] = game.ball.y
        obs[6] = -game.ball.vx; 
        obs[7] = game.ball.vy
        
        obs[8] = -game.agent_right.x; 
        obs[9] = game.agent_right.y
        obs[10] = -game.agent_right.vx; 
        obs[11] = game.agent_right.vy
        
        return obs

# ==============================================================================
# 2. MANAGER CALLBACK (VARIABLE RHYTHM)
# ==============================================================================
class LeagueManagerCallback(BaseCallback):
    def __init__(self, save_dir, best_model_path, 
                 warmup_steps, 
                 warmup_interval,  # Interval during warmup
                 league_interval,  # Interval during league
                 verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.best_model_path = best_model_path
        
        self.warmup_steps = warmup_steps
        self.warmup_interval = warmup_interval
        self.league_interval = league_interval
        
        self.generation = 0
        self.league_active = False
        self.next_update_target = self.warmup_interval 
        
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # A) Check league activation
        if not self.league_active and self.num_timesteps >= self.warmup_steps:
            self.league_active = True
            if self.verbose > 0:
                print(f"\n[LEAGUE] Warmup finished ({self.num_timesteps} steps). League opened.")
            self.training_env.env_method("activate_league")

        # B) Opponent generation
        if self.num_timesteps >= self.next_update_target:
            self.generation += 1
            
            target_name = f"gen_{self.generation}.zip"
            target_path = os.path.join(self.save_dir, target_name)
            
            if os.path.exists(self.best_model_path):
                if self.verbose > 0:
                    print(f"\n[LEAGUE] Step {self.num_timesteps}: Saving 'Best Model' to history.")
                shutil.copy(self.best_model_path, target_path)
                self.training_env.env_method("add_new_opponent", target_path)
            else:
                if self.verbose > 0:
                    print(f"WARNING: best_model.zip does not exist yet. Skipping save.")

            # Calculate next target
            if self.num_timesteps < self.warmup_steps:
                self.next_update_target += self.warmup_interval
            else:
                self.next_update_target += self.league_interval
            
            if self.verbose > 0:
                print(f"   --> Next update scheduled for step: {self.next_update_target}")

        return True

# ==============================================================================
# 3. TRAINING FUNCTION
# ==============================================================================
def train_phase4_variable(total_timesteps, n_envs, experiment_name, phase3_exp_name, algo="PPO"):
    
    final_exp_name = f"{experiment_name}_{algo.lower()}"
    log_dir, models_dir = setup_directories(experiment_name, "CL_Phase4", algo)
    history_dir = os.path.join(models_dir, "league_history")
    
    # Key paths
    best_model_path = os.path.join(models_dir, "best_model.zip")
    phase3_best = f"./model_logs/{algo.lower()}/{phase3_exp_name}/best_model.zip"

    print(f"--- PHASE 4: VARIABLE LEAGUE ({algo}) ---")
    print(f"• Warmup (0-600k): Interval 300k")
    print(f"• League (>600k): Interval 400k")

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
        n_envs = 1 # DQN prefers 1 env
        algo_kwargs = {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 0.1,
            "exploration_final_eps": 0.05
        }
    
    # Initial History
    if os.path.exists(phase3_best):
        os.makedirs(history_dir, exist_ok=True)
        shutil.copy(phase3_best, os.path.join(history_dir, "gen_0_phase3.zip"))
        print("History initialized with Phase 3 Champion.")
    else:
        print(f"Phase 3 not found at {phase3_best}. History empty.")
    
    # Environment
    def make_env():
        env = slimevolley.SlimeVolleyEnv()
        env = Phase4Wrapper(env, history_dir, AlgoClass)
        
        if algo == "DQN":
            env = DiscreteWrapper(env)
        return env

    env = make_vec_env(make_env, n_envs=n_envs)

    # Callbacks
    def make_eval_env():
        e = slimevolley.SlimeVolleyEnv()
        if algo == "DQN": 
            e = DiscreteWrapper(e)
        return e

    eval_callback = EvalCallback(
        make_eval_env(),
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=50000, 
        deterministic=True
    )
    
    league_callback = LeagueManagerCallback(
        save_dir=history_dir,
        best_model_path=best_model_path,
        warmup_steps=60000,
        warmup_interval=300000,
        league_interval=400000
    )

    # Load Agent
    start_model = phase3_best
    if os.path.exists(start_model):
        print(f"Loading previous model: {start_model}")
        model = AlgoClass.load(start_model, env=env, tensorboard_log=log_dir, **algo_kwargs)
    else:
        print("Creating agent from scratch.")
        model = AlgoClass("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, **algo_kwargs)

    # Train
    model.learn(total_timesteps=total_timesteps, 
                callback=[eval_callback, league_callback],
                tb_log_name=final_exp_name,
                reset_num_timesteps=True)
    
    model.save(os.path.join(models_dir, "final_champion"))
    env.close()

# ==============================================================================
# 4. WATCH FUNCTION (VISUALIZATION)
# ==============================================================================
def watch_phase4(exp_name, algo="PPO"):
    """
    Loads best Phase4 model vs basleine policy.
    Standard env used (agent right, baseline left).
    """
    model_path = os.path.join("./model_logs", f"{algo.lower()}/{exp_name}", "best_model.zip")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"LOADING MATCH ({algo}): {exp_name} (Right) vs BASELINE (Left)")
    
    # Load model
    if algo == "PPO": 
        AlgoClass = PPO
    elif algo == "DQN": 
        AlgoClass = DQN

    model = AlgoClass.load(model_path)
    
    # Create standard env
    env = slimevolley.SlimeVolleyEnv(render_mode="human")
    if algo == "DQN":
        env = DiscreteWrapper(env)
    
    obs = env.reset()
    total_reward = 0
    matches_played = 0
    agent_wins = 0
    
    try:
        while True:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                game = env.unwrapped.game 
                
                score_agent = 5 - game.agent_left.life
                score_opp = 5 - game.agent_right.life
                
                matches_played += 1
                if score_agent > score_opp:
                    agent_wins += 1
                    outcome = "VICTORY"
                elif score_agent > score_opp:
                    outcome = "TIE"
                else:
                    outcome = "LOSS"

                print(f"Match {matches_played} | {outcome} | Total reward: {total_reward} | Agent: {score_agent} - Opponent: {score_opp} | Win Rate: {agent_wins/matches_played:.1%}")
                obs = env.reset()
                total_reward = 0
                time.sleep(0.5)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nClosing visualization.")
        env.close()

# ==============================================================================
# 5. MAIN BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    MODE = "WATCH" 
    ALGO = "DQN"  
    
    CURRENT_EXP_NAME = "phase4"
    PHASE3_EXP_NAME = "phase3" 
    
    TOTAL_STEPS = 5000000
    N_ENVS = 4            

    if MODE == "TRAIN":
        train_phase4_variable(
            total_timesteps=TOTAL_STEPS,
            n_envs=N_ENVS,
            experiment_name=CURRENT_EXP_NAME,
            phase3_exp_name=PHASE3_EXP_NAME,
            algo=ALGO
        )
        
    elif MODE == "WATCH":
        watch_phase4(exp_name=CURRENT_EXP_NAME, algo=ALGO)