import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
from agri_env import AgriVoltaicEnv
import os

# 1. Load the trained agent
# We check which file exists to avoid "File Not Found" errors
model_name = "ppo_agri_agent_v2" 
if not os.path.exists(model_name + ".zip"):
    # Fallback to the default name if v2 doesn't exist
    model_name = "ppo_agri_agent"

print(f"Loading model: {model_name}")
env = DummyVecEnv([lambda: AgriVoltaicEnv()])
model = PPO.load(model_name)

# 2. Run a full day simulation
obs = env.reset()
sun_angles = []
panel_angles = []
rewards = []

print("Collecting data for visualization...")

for i in range(12):
    # Record the current Sun Angle BEFORE stepping
    current_sun = env.envs[0].state[0]
    sun_angles.append(current_sun)
    
    # Predict action
    action, _ = model.predict(obs)
    
    # --- THE FIX IS HERE ---
    # We use .item() to convert the numpy array [3] into the integer 3
    action_idx = action[0].item() 
    
    # Record the Panel Angle the agent chose
    angle_map = {0: 0.0, 1: 22.5, 2: 45.0, 3: 67.5, 4: 90.0}
    chosen_angle = angle_map[action_idx]
    panel_angles.append(chosen_angle)
    
    # Step
    obs, reward, done, info = env.step(action)
    rewards.append(reward[0])

# 3. Plot the Graph
hours = range(6, 18) # 6 AM to 6 PM

plt.figure(figsize=(10, 6))

# Plot Sun (Orange Dotted Line)
plt.plot(hours, sun_angles, label='Sun Position', color='orange', linestyle='--', marker='o')

# Plot Agent (Blue Line)
plt.plot(hours, panel_angles, label='AI Panel Angle', color='blue', linewidth=3, marker='x')

plt.title('Agri-Voltaic AI: Tracking the Sun', fontsize=14)
plt.xlabel('Time of Day (Hour)', fontsize=12)
plt.ylabel('Angle (Degrees)', fontsize=12)
plt.ylim(-5, 190) # Set limits to make the graph look neat
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('agri_voltaic_result.png')
print("Graph saved as 'agri_voltaic_result.png'. Check your project folder!")
plt.show()