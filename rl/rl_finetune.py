import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Bernoulli

# Add project root to path so we can import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.base_model import BaseModel
from rl.circle_env import CircleEnv

import yaml

def run_rl_finetuning():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load config
    config_path = os.path.join(project_root, "config", "train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["rl"]
        
    full_model_path = os.path.join(project_root, config["model_path"])
    
    # 1. Load the pre-trained Supervised Model
    model = BaseModel()
    try:
        model.load_state_dict(torch.load(full_model_path))
        print(f"Successfully loaded pre-trained Supervised Model from {full_model_path}")
    except FileNotFoundError:
        print(f"Error: Supervised model checkpoint not found at '{full_model_path}'. Run main.py first.")
        return

    # Use Adam optimizer for Policy Gradient
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # History logs for plotting
    reward_history = []
    running_avg_rewards = []
    
    print("\nStarting Reinforcement Learning (Policy Gradient - REINFORCE) Fine-tuning...")
    print("State: [x, y, r] | Action: 0 (OUTSIDE) or 1 (INSIDE) | Reward: +1.0 for Correct, -1.0 for Incorrect")
    print("="*80)
    
    # RL Environment parameters
    batch_size = config["batch_size"]

    
    # Initialize Gymnasium environment
    env = CircleEnv()
    
    epochs = config["epochs"]
    for epoch in range(1, epochs + 1):
        model.train()

        # We collect a batch of experiences
        batch_loss = 0.0
        epoch_rewards = []
        
        for _ in range(batch_size):
            # 1. Reset environment to get a new state/observation
            obs, info = env.reset()
            state = torch.tensor(obs, dtype=torch.float32)

            # 2. Get policy probability (probability of predicting INSIDE)
            prob = model(state.unsqueeze(0)) # shape: [1, 1]
            
            # 3. Sample action from Policy distribution (Bernoulli)
            dist = Bernoulli(prob)
            action = dist.sample() # 1.0 (INSIDE) or 0.0 (OUTSIDE)
            
            # 4. Environment step: Pass the sampled action into the env
            # action.item() is float, convert to int for discrete space
            next_state, reward, terminated, truncated, env_info = env.step(int(action.item()))
                
            epoch_rewards.append(reward)
            
            # 5. Compute Policy Gradient Loss: -log_prob * Reward
            log_prob = dist.log_prob(action)
            loss = -log_prob * reward
            batch_loss += loss
            
        # 6. Optimize Policy Network weights
        optimizer.zero_grad()
        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        optimizer.step()
        
        # Track rewards
        avg_reward = np.mean(epoch_rewards)
        reward_history.append(avg_reward)
        
        # Calculate running average of rewards
        running_avg = np.mean(reward_history[-50:])
        running_avg_rewards.append(running_avg)
        
        if epoch % 100 == 0 or epoch == 1:
            print(f"RL Epoch {epoch:4d}/{epochs} | Batch Loss: {batch_loss.item():.4f} | Avg Batch Reward: {avg_reward:+.3f} | Running Avg Reward (last 50): {running_avg:+.3f}")

    print("="*80)
    
    # 7. Save the RL fine-tuned model checkpoint inside output/ directory
    rl_model_path = os.path.join(project_root, config["rl_model_save_path"])
    os.makedirs(os.path.dirname(rl_model_path), exist_ok=True)
    torch.save(model.state_dict(), rl_model_path)
    print(f"RL Fine-tuned Model saved to: {rl_model_path}")

    
    # 8. Plot training rewards progress
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Average Epoch Reward', color='skyblue', alpha=0.5)
    plt.plot(running_avg_rewards, label='Running Avg Reward (Window=50)', color='darkblue', linewidth=2)
    plt.axhline(1.0, color='green', linestyle='--', label='Perfect Reward (1.0)')
    plt.axhline(0.0, color='gray', linestyle='-', linewidth=0.8)
    plt.title("Reinforcement Learning Fine-tuning Progress (REINFORCE)")
    plt.xlabel("RL Epochs")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot
    plots_dir = os.path.join(project_root, "evaluate", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_save_path = os.path.join(plots_dir, "rl_training_rewards.png")
    plt.savefig(plot_save_path, dpi=150)
    print(f"RL Reward progress plot saved to: {plot_save_path}\n")
    plt.close()

if __name__ == "__main__":
    run_rl_finetuning()
