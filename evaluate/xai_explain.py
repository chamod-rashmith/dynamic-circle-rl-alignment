import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path so we can import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.base_model import BaseModel

def compute_integrated_gradients(model, input_tensor, baseline_tensor, steps=50):
    """
    Computes Integrated Gradients for a single input relative to a baseline.
    Integrated Gradients is an industry-standard XAI method for neural networks.
    """
    # Generate scaled inputs along the path
    alphas = np.linspace(0.0, 1.0, steps + 1)
    scaled_inputs = []
    for alpha in alphas:
        scaled_input = baseline_tensor + alpha * (input_tensor - baseline_tensor)
        scaled_inputs.append(scaled_input)
        
    scaled_inputs_tensor = torch.cat(scaled_inputs, dim=0).requires_grad_(True)
    
    # Forward pass
    outputs = model(scaled_inputs_tensor)
    
    # Compute gradients of the output with respect to scaled inputs
    grads = torch.autograd.grad(outputs.sum(), scaled_inputs_tensor)[0]
    
    # Approximate the integral using trapezoidal rule
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    mean_grads = torch.mean(avg_grads, dim=0, keepdim=True)
    
    # Integrated Gradients = (input - baseline) * mean_gradient
    integrated_grad = (input_tensor - baseline_tensor) * mean_grads
    return integrated_grad.squeeze().detach().cpu().numpy()

def explain_model(x_val=0.4, y_val=0.4, r_val=0.7, model_path="output/dynamic_radius_model.pth"):
    # Resolve relative path against project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_model_path = os.path.join(project_root, model_path)
    
    # 1. Load model
    model = BaseModel()
    try:
        model.load_state_dict(torch.load(full_model_path))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at '{full_model_path}'. Run main.py first.")
        return

    # 2. Local Explanation: Integrated Gradients for a specific point
    input_tensor = torch.tensor([[x_val, y_val, r_val]], dtype=torch.float32)
    baseline_tensor = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32) # Zero baseline
    
    ig_attributions = compute_integrated_gradients(model, input_tensor, baseline_tensor)
    
    # 3. Global Explanation: Average gradient magnitude over a synthetic dataset
    # This shows which features the model relies on most on average.
    np.random.seed(42)
    n_samples = 500
    X_coords = np.random.uniform(-1.0, 1.0, (n_samples, 2))
    r = np.random.uniform(0.3, 0.9, (n_samples, 1))
    X_synthetic = np.hstack((X_coords, r))
    X_tensor = torch.tensor(X_synthetic, dtype=torch.float32).requires_grad_(True)
    
    outputs = model(X_tensor)
    grads = torch.autograd.grad(outputs.sum(), X_tensor)[0]
    global_importance = torch.mean(torch.abs(grads), dim=0).detach().cpu().numpy()
    
    # Print numerical results
    print("="*60)
    print("           EXPLAINABLE AI (XAI) REPORT (Captum-like IG)           ")
    print("="*60)
    print(f"Local Point Profile: X={x_val}, Y={y_val}, Radius={r_val}")
    distance = np.sqrt(x_val**2 + y_val**2)
    print(f"Distance from origin: {distance:.4f} vs Radius: {r_val}")
    with torch.no_grad():
        prob = model(input_tensor).item()
    print(f"Model Predicted Probability of being INSIDE: {prob * 100:.2f}%")
    print("-"*60)
    print("Local Feature Attribution (Integrated Gradients):")
    features = ['Input X', 'Input Y', 'Input R']
    for feat, attr in zip(features, ig_attributions):
        print(f"  {feat:10s} : Attribution = {attr:+.4f} ({'Supports INSIDE' if attr > 0 else 'Supports OUTSIDE'})")
    print("-"*60)
    print("Global Feature Importance (Mean Absolute Gradients):")
    for feat, imp in zip(features, global_importance):
        print(f"  {feat:10s} : Importance = {imp:.4f}")
    print("="*60 + "\n")

    # 4. Visualization Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Local Attribution (Water Fall / Bar Chart style)
    colors = ['green' if val >= 0 else 'red' for val in ig_attributions]
    axes[0].barh(features, ig_attributions, color=colors, edgecolor='black', height=0.5)
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_title(f"Local Feature Attribution (IG) for Pt ({x_val}, {y_val}) r={r_val}")
    axes[0].set_xlabel("Attribution Value (Supports INSIDE (+) / OUTSIDE (-))")
    axes[0].grid(True, linestyle=':', alpha=0.5)
    
    # Annotate attribution values on bars
    for idx, val in enumerate(ig_attributions):
        axes[0].text(val + (0.01 if val >= 0 else -0.05), idx, f"{val:+.3f}", va='center', weight='bold')

    # Plot 2: Global Feature Importance
    axes[1].bar(features, global_importance, color='darkorange', edgecolor='black', width=0.4)
    axes[1].set_title("Global Feature Importance (Mean Absolute Gradients)")
    axes[1].set_ylabel("Importance (Sensitivity)")
    axes[1].grid(True, linestyle=':', alpha=0.5)
    
    for idx, val in enumerate(global_importance):
        axes[1].text(idx, val + 0.01, f"{val:.3f}", ha='center', va='bottom', weight='bold')

    plt.suptitle("Explainable AI (XAI) - BaseModel Interpretability Report", fontsize=15)
    plt.tight_layout()
    
    # Save in evaluate/plots/ directory
    plots_dir = os.path.join(project_root, "evaluate", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    xai_plot_path = os.path.join(plots_dir, "xai_explanation.png")
    plt.savefig(xai_plot_path, dpi=150)
    print(f"XAI explanation plot saved to: {xai_plot_path}")
    plt.close()

if __name__ == "__main__":
    # Test point (0.4, 0.4) with radius 0.7
    explain_model(x_val=0.4, y_val=0.4, r_val=0.7)
