import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path so we can import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.base_model import BaseModel

def plot_model_weights_and_stats(model_path="output/dynamic_radius_model.pth"):
    # Resolve relative path against project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_model_path = os.path.join(project_root, model_path)
    
    # 1. Load model
    model = BaseModel()
    try:
        model.load_state_dict(torch.load(full_model_path))
        model.eval()
        print(f"Loaded model successfully from {full_model_path}\n")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at '{full_model_path}'. Run main.py first.")
        return

    # 2. Extract and Print Statistics
    print("="*50)
    print("                MODEL STATISTICS                ")
    print("="*50)
    
    total_params = 0
    params_to_plot = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            param_np = param.data.cpu().numpy()
            params_to_plot.append((name, param_np))
            
            print(f"Layer: {name:20s} | Shape: {str(list(param.shape)):15s} | Parameters: {num_params:4d}")
            print(f"  -> Mean: {param_np.mean():.4f} | Std: {param_np.std():.4f} | Min: {param_np.min():.4f} | Max: {param_np.max():.4f}")
            print("-"*50)
            
    print(f"Total Trainable Parameters: {total_params}")
    print("="*50 + "\n")

    # 3. Dynamic Plotting (Histograms for all parameters)
    n_params = len(params_to_plot)
    cols = 2
    rows = (n_params + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()
    
    for i, (name, param_np) in enumerate(params_to_plot):
        ax = axes[i]
        ax.hist(param_np.ravel(), bins=20, color='royalblue' if 'weight' in name else 'crimson', edgecolor='black', alpha=0.7)
        ax.set_title(f"{name} Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle=':', alpha=0.6)
        
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("BaseModel Trained Weights & Biases Analysis", fontsize=14)
    plt.tight_layout()
    
    # Save in evaluate/plots/ directory
    plots_dir = os.path.join(project_root, "evaluate", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_save_path = os.path.join(plots_dir, "model_weight_stats.png")
    plt.savefig(plot_save_path, dpi=150)
    print(f"Weight analysis plot successfully saved to: {plot_save_path}")
    plt.close()

if __name__ == "__main__":
    plot_model_weights_and_stats()
