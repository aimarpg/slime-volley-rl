import time
import os
import slimevolley
from stable_baselines3 import PPO

def test_vs_baseline(model_name="phase1"):
    path = f"./model_logs/{model_name}/best_model"
    
    if not os.path.exists(path + ".zip"):
        print(f"File not found: {path}.zip")
        return

    print(f"Main Event: {model_name} (Agent) vs Baseline")
    
    env = slimevolley.SlimeVolleyEnv()
    model = PPO.load(path)
    
    obs = env.reset()
    done = False
    total_reward = 0
    
    try:
        while True:
            env.render()
            
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            if done:
                print(f"Match finished. Episode score: {total_reward}")
                obs = env.reset()
                total_reward = 0
                time.sleep(1.0)
            
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    MODEL_TO_TEST = "phase3" 
    test_vs_baseline(MODEL_TO_TEST)