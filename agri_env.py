import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AgriVoltaicEnv(gym.Env):
    def __init__(self):
        super(AgriVoltaicEnv, self).__init__()
        
        # 0=Flat, 1=22deg, 2=45deg, 3=67deg, 4=Vertical
        self.action_space = spaces.Discrete(5)
        
        self.action_to_angle = {
            0: 0.0,
            1: 22.5,
            2: 45.0,
            3: 67.5,
            4: 90.0
        }

        # Observation: [Sun Angle, Moisture, Plant Health]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([180, 100, 100]), 
            dtype=np.float32
        )

        self.state = None
        self.current_hour = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 100.0, 0.0], dtype=np.float32)
        self.current_hour = 6
        return self.state, {}

    def step(self, action):
        sun_angle, moisture, crop_progress = self.state
        
        # --- THE FIX IS HERE ---
        # We force the action to be a standard integer using int()
        # This handles cases where the AI sends a numpy array
        val = int(action)
        panel_angle = self.action_to_angle[val]

        # --- PHYSICS ---
        
        # 1. Power Logic
        angle_diff = abs(sun_angle - panel_angle)
        efficiency = max(0, 1.0 - (angle_diff / 45.0)) 
        power_generated = 10.0 * efficiency

        # 2. Crop Logic
        light_intensity = (panel_angle / 90.0)
        
        growth = 0
        if moisture > 10:
            growth = light_intensity * 1.0
            moisture -= 2

        # --- REWARD ---
        reward = (power_generated * 1.0) + (growth * 2.0)

        # --- TIME ---
        self.current_hour += 1
        sun_angle += 15 
        
        terminated = False
        if self.current_hour >= 18:
            terminated = True

        self.state = np.array([sun_angle, moisture, crop_progress], dtype=np.float32)
        
        info = {"panel_angle": panel_angle}
        
        return self.state, reward, terminated, False, info