import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path so we can import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.base_model import BaseModel

def compare_checkpoints(supervised_path="output/dynamic_radius_model.pth", rl_path="output/rl_finetuned_model.pth"):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_sup_path = os.path.join(project_root, supervised_path)
    full_rl_path = os.path.join(project_root, rl_path)
    
    # 1. Load models
    model_sup = BaseModel()
    model_rl = BaseModel()
    
    try:
        model_sup.load_state_dict(torch.load(full_sup_path))
        model_rl.load_state_dict(torch.load(full_rl_path))
        print("Successfully loaded both Supervised and RL models.")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    # Extract state dicts
    sd_sup = model_sup.state_dict()
    sd_rl = model_rl.state_dict()
    
    print("\n" + "="*70)
    print("             WEIGHT DIFFERENCE & DRIFT REPORT             ")
    print("="*70)
    
    params_to_compare = []
    
    # Compare each parameter
    for name in sd_sup.keys():
        w_sup = sd_sup[name].cpu().numpy()
        w_rl = sd_rl[name].cpu().numpy()
        
        abs_diff = np.abs(w_sup - w_rl)
        mean_diff = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
        mse = np.mean((w_sup - w_rl) ** 2)
        
        params_to_compare.append((name, w_sup, w_rl, mean_diff))
        
        print(f"Parameter: {name:20s} | Shape: {str(list(w_sup.shape)):10s}")
        print(f"  -> Mean Abs Change: {mean_diff:.4f} | Max Change: {max_diff:.4f} | MSE: {mse:.4f}")
        print(f"  -> Supervised Mean : {w_sup.mean():.4f} | RL Mean : {w_rl.mean():.4f}")
        print("-"*70)
        
    # 2. Draw Plot Comparison (Dynamic Histograms side-by-side)
    n_params = len(params_to_compare)
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3.5 * n_params))
    
    for i, (name, w_sup, w_rl, mean_diff) in enumerate(params_to_compare):
        # Supervised Histogram
        axes[i, 0].hist(w_sup.ravel(), bins=20, color='dodgerblue', edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f"Supervised: {name}")
        axes[i, 0].set_ylabel("Frequency")
        axes[i, 0].grid(True, linestyle=':', alpha=0.5)
        
        # RL Histogram
        axes[i, 1].hist(w_rl.ravel(), bins=20, color='tomato', edgecolor='black', alpha=0.7)
        axes[i, 1].set_title(f"RL Fine-Tuned: {name} (Mean Diff: {mean_diff:.4f})")
        axes[i, 1].grid(True, linestyle=':', alpha=0.5)

    plt.suptitle("Weight Distributions: Supervised Learning vs RL Fine-Tuning", fontsize=14)
    plt.tight_layout()
    
    plots_dir = os.path.join(project_root, "evaluate", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_save_path = os.path.join(plots_dir, "weight_comparison.png")
    plt.savefig(plot_save_path, dpi=150)
    print(f"\nComparison plot successfully saved to: {plot_save_path}\n")
    plt.close()

if __name__ == "__main__":
    compare_checkpoints()
