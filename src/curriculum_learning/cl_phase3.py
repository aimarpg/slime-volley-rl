import sys
import os
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../..")))

import gym
import time
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from custom_slimevolleygym import slimevolley0v1 as slimevolley
from cl_helpers import RenderCallback, setup_directories, DiscreteWrapper

# ==============================================================================
# 1. PHASE 3 WRAPPER (iterative opponent update)
# ==============================================================================
class Phase3Wrapper(gym.Wrapper):
    """
    Wrapper to control the left agent (Opponent).
    - Mirrors observation (inverts X).
    - Predicts opponent action.
    - BRIDGING: Converts DQN integer actions to vectors if necessary.
    """
    def __init__(self, env, opponent_model):
        super().__init__(env)
        self.opponent_model = opponent_model
        self.prev_dist = 0.0

    def reset(self):
        obs = self.env.reset()
        game = self.env.unwrapped.game
        self.prev_dist = self._get_distance(game)
        return obs

    def step(self, action):
        game = self.env.unwrapped.game
        
        # 1. Generate mirrored observation
        opp_obs = self._get_opponent_observation(game)
        
        # 2. Predict opponent action
        try:
            raw_action, _ = self.opponent_model.predict(opp_obs, deterministic=False)
            
            if isinstance(raw_action, tuple): 
                    raw_action = raw_action[0]

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
        
        # 3. Environment Step
        obs, reward, done, info = self.env.unwrapped.step(action, otherAction=opponent_action)
        
        return obs, reward, done, info

    def update_opponent(self, new_model):
        """ Updates the opponent model object. """
        self.opponent_model = new_model

    def _get_distance(self, game):
        dx = game.ball.x - game.agent_right.x
        dy = game.ball.y - game.agent_right.y
        return np.sqrt(dx*dx + dy*dy)

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
# 2. OPPONENT UPDATE CALLBACK
# ==============================================================================
class SelfPlayUpdateCallback(BaseCallback):
    """
    Saves current model and reloads it as the new opponent.
    Requires `AlgoClass` to know whether to load as PPO or DQN.
    """
    def __init__(self, update_interval, save_dir, AlgoClass, verbose=1):
        super().__init__(verbose)
        self.update_interval = update_interval
        self.save_dir = save_dir
        self.AlgoClass = AlgoClass # Store class reference (PPO or DQN)
        self.generation = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.update_interval == 0:
            self.generation += 1
            gen_name = f"opponent_gen_{self.generation}"
            save_path = os.path.join(self.save_dir, gen_name)
            
            if self.verbose > 0:
                print(f"\n[SELF-PLAY] Gen {self.generation}: Updating opponent ({self.AlgoClass.__name__}).")

            # 1. Save current model
            self.model.save(save_path)
            
            # 2. Load as inference object
            new_opponent = self.AlgoClass.load(save_path, device="cpu")
            
            # 3. Inject into environment
            self.training_env.env_method("update_opponent", new_opponent)
            
        return True

# ==============================================================================
# 3. TRAINING FUNCTION
# ==============================================================================
def train_phase3(total_timesteps=2000000, n_envs=4, render=False, 
                 experiment_name="phase3", previous_model="phase2", algo="PPO"):
    
    final_exp_name = f"{experiment_name}_{algo.lower()}"
    log_dir, models_dir = setup_directories(experiment_name, "CL_Phase3", algo)
    history_dir = os.path.join(models_dir, "opponent_history")

    print(f"--- PHASE 3: ITERATIVE SELF-PLAY ({algo}) ---")

    # Algorithm configurations
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

    # 1. LOAD INITIAL OPPONENT (Best of Phase 2)
    opponent_path = os.path.join("./model_logs", f"{algo.lower()}/{previous_model}", "best_model.zip")
    
    if not os.path.exists(opponent_path):
        print(f"ERROR: Previous model not found at {opponent_path}")
        return

    print(f"Initial Opponent: {opponent_path}")
    initial_opponent = AlgoClass.load(opponent_path, device="cpu")

    # 2. CONFIGURE ENVIRONMENT
    if render:
        print("Render enabled: Forcing n_envs=1.")
        n_envs = 1

    def make_env():
        env = slimevolley.SlimeVolleyEnv()
        env = Phase3Wrapper(env, initial_opponent)
        
        if algo == "DQN":
            env = DiscreteWrapper(env)

        return env

    env = make_vec_env(make_env, n_envs=n_envs)
    eval_env = make_env() 

    # 3. CALLBACKS
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=20000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
     # Update opponent during training
    update_freq = 60000 
    
    self_play_callback = SelfPlayUpdateCallback(
        update_interval=update_freq,
        save_dir=history_dir,
        AlgoClass=AlgoClass 
    )

    callbacks = [eval_callback, self_play_callback]
    if render:
        callbacks.append(RenderCallback(True))

    # 4. INITIALIZE AGENT (Transfer Learning)
    print(f"Loading Phase 2 brain ({algo}) to continue learning...")
    model = AlgoClass.load(opponent_path, env=env, 
                           tensorboard_log=log_dir,
                           **algo_kwargs)

    # 5. TRAIN
    try:
        model.learn(total_timesteps=total_timesteps, 
                    callback=callbacks, 
                    tb_log_name=final_exp_name,
                    reset_num_timesteps=True)
        
        print("Phase 3 Training finished.")
        model.save(os.path.join(models_dir, "final_model"))

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        model.save(os.path.join(models_dir, "interrupted_model"))
    finally:
        env.close()
        eval_env.close()

# ==============================================================================
# 4. WATCH FUNCTION
# ==============================================================================
def watch_phase3(agent_name="phase3", opponent_name="phase2", algo="PPO"):
    """
    Visualizes a match between the Phase 3 model and a previous model.
    """
    # Construct paths based on algo structure
    agent_path = os.path.join("./model_logs", f"{algo.lower()}/{agent_name}", "best_model.zip")
    opp_path = os.path.join("./model_logs", f"{algo.lower()}/{opponent_name}", "best_model.zip")
    
    if not os.path.exists(agent_path) or not os.path.exists(opp_path):
        print(f"Missing models.\nAgent: {agent_path}\nOpponent: {opp_path}")
        return

    print(f"MATCH ({algo}): {agent_name} (Right) vs {opponent_name} (Left)")
    
    if algo == "PPO": 
        AlgoClass = PPO
    elif algo == "DQN": 
        AlgoClass = DQN

    agent_model = AlgoClass.load(agent_path)
    opp_model = AlgoClass.load(opp_path)

    env = slimevolley.SlimeVolleyEnv(render_mode="human")
    
    env = Phase3Wrapper(env, opp_model)
    if algo == "DQN":
        env = DiscreteWrapper(env)

    obs = env.reset()
    total_reward = 0
    matches_played = 0
    
    try:
        while True:
            env.render()
            action, _ = agent_model.predict(obs, deterministic=True)
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
            
            time.sleep(0.02)
    except KeyboardInterrupt:
        env.close()

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":

    MODE = "WATCH" 
    ALGO = "DQN"
    
    EXP_NAME_PHASE_3 = "phase3"
    EXP_NAME_PHASE_2 = "phase2" 
    
    TOTAL_STEPS = 3000000 
    N_ENVS = 4 

    if MODE == "TRAIN":
        train_phase3(total_timesteps=TOTAL_STEPS, 
                     n_envs=N_ENVS, 
                     render=False,
                     experiment_name=EXP_NAME_PHASE_3,
                     previous_model=EXP_NAME_PHASE_2,
                     algo=ALGO)
                     
    elif MODE == "WATCH":
        watch_phase3(agent_name=EXP_NAME_PHASE_3, 
                     opponent_name=EXP_NAME_PHASE_2,
                     algo=ALGO)