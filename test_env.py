from agri_env import AgriVoltaicEnv

env = AgriVoltaicEnv()
obs, _ = env.reset()

print("--- Starting Simulation ---")
print(f"Initial State: {obs}")

done = False
while not done:
    # Random action: Pick a random angle between 0 and 90
    action = env.action_space.sample()
    
    # Step the environment
    obs, reward, done, _, _ = env.step([action])
    
    print(f"Action: {action[0]:.2f}° | Reward: ${reward:.2f} | New State: {obs}")

print("--- Day Over ---")