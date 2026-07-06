import os
import time
import gym
from gym import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from custom_slimevolleygym import slimevolley0v1 as slimevolley

# ==========================================
# 1. DISCRETE WRAPPER
# ==========================================
class DiscreteWrapper(gym.ActionWrapper):
    """
    Adapter: Converts SlimeVolley's MultiBinary(3) to Discrete(6).
    Required for DQN.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(6)
        self.mapping = {
            0: [0, 0, 0], # NOOP
            1: [1, 0, 0], # LEFT (Forward)
            2: [0, 1, 0], # RIGHT (Backward)
            3: [0, 0, 1], # JUMP
            4: [1, 0, 1], # LEFT + JUMP
            5: [0, 1, 1], # RIGHT + JUMP
        }

    def action(self, action):
        if hasattr(action, "item"):
            action = action.item()
        return self.mapping[int(action)]

    @staticmethod
    def to_vector(action_index):
        """Helper to convert scalar index back to vector for Opponents"""
        mapping = {
            0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0],
            3: [0, 0, 1], 4: [1, 0, 1], 5: [0, 1, 1]
        }
        
        if hasattr(action_index, 'item'):
            action_index = action_index.item()
            
        return mapping.get(int(action_index), [0, 0, 0])

# ==========================================
# 2. DIRECTORY MANAGEMENT
# ==========================================
def setup_directories(experiment_name, phase_folder_name, algo):
    """
    Creates and returns paths for tensorboard logs and model storage.
    """
    log_dir = os.path.join("./tensorboard_logs", phase_folder_name)
    models_dir = os.path.join("./model_logs", algo.lower(), experiment_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    return log_dir, models_dir

# ==========================================
# 3. RENDER CALLBACK
# ==========================================
class RenderCallback(BaseCallback):
    def __init__(self, render_enabled=False, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_enabled = render_enabled

    def _on_step(self) -> bool:
        if self.render_enabled:
            self.training_env.render()
        return True

# ==========================================
# 4. GENERIC VISUALIZATION
# ==========================================
def watch(model_path, wrapper_class=None, algo="PPO"):
    """
    Loads a model and runs a visualization loop.
    
    :param model_path: Path to the .zip file.
    :param wrapper_class: (Optional) Logic Wrapper class (e.g., Phase1Wrapper).
    :param algo: "PPO" or "DQN".
    """
    if not model_path.endswith(".zip"):
        model_path += ".zip"

    if not os.path.exists(model_path):
        print(f"Error: File {model_path} not found.")
        return

    print(f"Loading {algo} model: {model_path}")
    
    if algo == "PPO":
        AlgoClass = PPO
    elif algo == "DQN":
        AlgoClass = DQN
    else:
        print(f"Unknown algorithm: {algo}")
        return

    env = slimevolley.SlimeVolleyEnv(render_mode="human")
    
    if wrapper_class is not None:
        print(f"Applying Logic Wrapper: {wrapper_class.__name__}")
        env = wrapper_class(env)
        
    if algo == "DQN":
        print("Applying Discrete Wrapper for DQN")
        env = DiscreteWrapper(env)

    model = AlgoClass.load(model_path)

    matches_played = 0
    agent_wins = 0
    
    obs = env.reset()
    total_reward = 0

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
            
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
        env.close()