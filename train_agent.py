import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from agri_env import AgriVoltaicEnv

# 1. Setup Environment
env = DummyVecEnv([lambda: AgriVoltaicEnv()])

# 2. Define Model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)

print("--- Training Discrete Agent ---")
model.learn(total_timesteps=30000) 

print("--- Training Finished ---")

# --- SAVE THE MODEL (This was missing before!) ---
model.save("ppo_agri_agent_v2")
print("New Discrete Model saved as 'ppo_agri_agent_v2.zip'")

# --- EVALUATION ---
obs = env.reset()
print("\n--- Testing the Discrete Agent ---")

for i in range(12):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    real_angle = info[0]["panel_angle"]
    sun_val = obs[0][0] 
    
    print(f"Sun: {sun_val:.0f}° | Choice: {action[0]} ({real_angle}°)| Reward: {reward[0]:.2f}")