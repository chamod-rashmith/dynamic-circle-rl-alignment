import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CircleEnv(gym.Env):
    """
    Gymnasium environment for the Dynamic 2D Circle Classifier.
    State (Observation): [x, y, r]
      - x, y: coordinates in [-15.0, 15.0]
      - r: circle radius in [0.5, 15.0]
    Action: 0 (OUTSIDE) or 1 (INSIDE)
    Reward: +1.0 for correct prediction, -1.5 for incorrect prediction
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super(CircleEnv, self).__init__()

        # State space (Observation space): [x, y, r]
        low = np.array([-15.0, -15.0, 0.5], dtype=np.float32)
        high = np.array([15.0, 15.0, 15.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: 0 (OUTSIDE) or 1 (INSIDE)
        self.action_space = spaces.Discrete(2)
        
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Dynamically generate a random state
        x = np.random.uniform(-15.0, 15.0)
        y = np.random.uniform(-15.0, 15.0)
        r = np.random.uniform(0.5, 15.0)
        
        self.state = np.array([x, y, r], dtype=np.float32)
        
        info = {}
        return self.state, info

    def step(self, action):
        # Extract features from current state
        x, y, r = self.state
        
        # Calculate ground truth distance from origin
        distance = np.sqrt(x**2 + y**2)
        actual_label = 1 if distance <= r else 0

        # Reward formulation
        if action == actual_label:
            reward = 1.0
        else:
            # Continuous penalty: larger penalty when far away from boundary, smaller penalty when close
            reward = -(0.5 + 0.2 * abs(distance - r))

        # Single-step episode
        terminated = True
        truncated = False
        
        info = {
            "correct": action == actual_label,
            "actual_label": actual_label,
            "boundary_distance": abs(distance - r)
        }

        return self.state, reward, terminated, truncated, info

if __name__ == "__main__":
    print("--- Testing CircleEnv (Gymnasium Environment) ---")
    env = CircleEnv()
    
    # Run 5 sample steps to demonstrate how it works
    for episode in range(1, 6):
        obs, info = env.reset()
        x, y, r = obs
        
        # Take a random action (0 or 1)
        action = env.action_space.sample()
        
        # Step through the environment
        next_obs, reward, terminated, truncated, env_info = env.step(action)
        
        action_name = "INSIDE (1)" if action == 1 else "OUTSIDE (0)"
        actual_name = "INSIDE (1)" if env_info["actual_label"] == 1 else "OUTSIDE (0)"
        result = "CORRECT" if env_info["correct"] else "INCORRECT"
        
        print(f"\n[Episode {episode}]")
        print(f"  State (Point): ({x:+.2f}, {y:+.2f}) | Radius (r): {r:.2f}")
        print(f"  Action Chosen: {action_name}")
        print(f"  Actual Target: {actual_name} -> {result}")
        print(f"  Reward Earned: {reward:+.1f}")

