import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.base_model import BaseModel
from train import train_model

def generate_multi_radius_data(n_samples=5000):
    """
    Generates 2D points with a dynamic radius 'r' as the 3rd feature.
    Inputs: [x, y, r]
    Label: 1 if distance from origin <= r, else 0
    """
    # 2D points in [-3.0, 3.0] x [-3.0, 3.0]
    X_coords = np.random.uniform(-3.0, 3.0, (n_samples, 2))
    
    # Random radius for each sample between 0.3 and 3.0
    r = np.random.uniform(0.3, 3.0, (n_samples, 1))
    
    # Calculate distances
    distances = np.sqrt(X_coords[:, 0]**2 + X_coords[:, 1]**2).reshape(-1, 1)
    
    # Label is 1 if distance <= radius, else 0
    y = (distances <= r).astype(np.float32)
    
    # Features: [x, y, r]
    X = np.hstack((X_coords, r))
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main():
    import yaml
    
    # Load config
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, "config", "train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["supervised"]
        
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Generate Dataset
    print("Generating dataset with dynamic radius input...")
    X, y = generate_multi_radius_data(n_samples=config["n_samples"])

    
    # Split into train (80%) and test/val (20%)
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # 2. Initialize Model
    model = BaseModel()
    
    # 3. Train using the separate train loop from train.py
    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config["epochs"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        patience=config["patience"]
    )
    
    # 4. Save trained model inside output/ directory
    model_save_path = os.path.join(config_dir, config["model_save_path"])
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    
    # 5. Visualizing the Decision Boundary for different values of 'r'
    print("Generating decision boundary plot for multiple radiuses...")
    radiuses_to_test = [0.5, 1.5, 2.5]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    xx, yy = np.meshgrid(np.linspace(-3.1, 3.1, 200), np.linspace(-3.1, 3.1, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    for i, r_val in enumerate(radiuses_to_test):
        ax = axes[i]
        
        # Create grid with the constant radius r_val as the 3rd feature
        r_col = np.full((len(grid_points), 1), r_val)
        grid_features = np.hstack((grid_points, r_col))
        grid_tensor = torch.tensor(grid_features, dtype=torch.float32)
        
        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(grid_tensor).reshape(xx.shape).numpy()
            
        # Draw contour at prediction probability 0.5 (Decision boundary)
        contour = ax.contour(xx, yy, preds, levels=[0.5], colors='green', linewidths=3)
        ax.clabel(contour, inline=True, fmt=f'Learned Boundary', fontsize=10)
        
        # Draw the true circle boundary
        true_circle = plt.Circle((0, 0), r_val, color='black', fill=False, linestyle='--', linewidth=2, label=f'True r={r_val}')
        ax.add_patch(true_circle)
        
        # Plot some validation points matching this specific radius (approximately)
        # Filter validation points that have radius close to r_val (+/- 0.1)
        mask = np.abs(X_val[:, 2].numpy() - r_val) < 0.1
        X_filtered = X_val[mask].numpy()
        y_filtered = y_val[mask].numpy().squeeze()
        
        if len(X_filtered) > 0:
            ax.scatter(X_filtered[y_filtered == 1, 0], X_filtered[y_filtered == 1, 1], c='blue', alpha=0.6, s=15, label='Inside')
            ax.scatter(X_filtered[y_filtered == 0, 0], X_filtered[y_filtered == 0, 1], c='red', alpha=0.6, s=15, label='Outside')
            
        ax.set_title(f"Radius (r) = {r_val}")
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-3.1, 3.1)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')

        
    plt.suptitle("BaseModel Decision Boundaries for Different Input Radiuses (r)")
    plt.tight_layout()
    
    # Save in evaluate/plots/ directory
    os.makedirs("evaluate/plots", exist_ok=True)
    plot_path = "evaluate/plots/multi_radius_decision_boundary.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Decision boundary plot saved as {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
    
