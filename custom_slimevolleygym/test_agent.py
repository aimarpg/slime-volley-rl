import time
import slimevolley

env = slimevolley.SlimeVolleyEnv()
my_policy = slimevolley.BaselinePolicy()

def run_baseline_agent():
    obs = env.reset()
    my_policy.reset()
    
    total_reward = 0
    done = False
    
    print("Match started: Baseline (Right) vs Baseline (Left)")
    
    while not done:
        env.render()
        
        action = my_policy.predict(obs)
        obs, reward, done, info = env.step(action)
        
        print(f"Observations: {obs}")
        game = env.unwrapped.game
        print(f"Left Agent (x,y): {game.agent_left.x}, {game.agent_left.y}; Right Agent (x,y): {game.agent_right.x}, {game.agent_right.y}")
        print(f"Ball (x,vx): {game.ball.x}, {game.ball.vx}")
        print(f"Info: {info.get('otherObs')}\n")
        
        total_reward += reward
        time.sleep(0.02)

    print(f"Game over. Total reward: {total_reward}")

if __name__ == "__main__":
    try:
        for i in range(3):
            print(f"--- Episode {i+1} ---")
            run_baseline_agent()
    finally:
        env.close()