import sys
import os
import torch
import numpy as np

# Add project root to path so we can import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.base_model import BaseModel

def check_point(x, y, r):
    # 1. Load the model
    model = BaseModel()
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "rl_finetuned_model.pth")
    
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at '{model_path}'. Run main.py first.")
        return

    # 2. Predict using the model
    input_tensor = torch.tensor([[x, y, r]], dtype=torch.float32)
    with torch.no_grad():
        probability = model(input_tensor).item()
        prediction = "INSIDE" if probability >= 0.5 else "OUTSIDE"

    # 3. Ground Truth (Mathematical calculation)
    distance = np.sqrt(x**2 + y**2)
    actual = "INSIDE" if distance <= r else "OUTSIDE"

    # 4. Print Results
    print(f"--- Checking Point ({x}, {y}) with Radius {r} ---")
    print(f"Distance from origin: {distance:.4f}")
    print(f"Ground Truth Math   : {actual}")
    print(f"Model Prediction    : {prediction} (Prob: {probability*100:.2f}%)")
    print(f"Correct?            : {'Yes' if prediction == actual else 'No'}\n")

if __name__ == "__main__":
    # >>> EDIT THESE VALUES TO TEST DIFFERENT POINTS <<<
    X_COORDINATE = 0.4
    Y_COORDINATE = 0.4
    RADIUS = 0.7

    
    check_point(X_COORDINATE, Y_COORDINATE, RADIUS)
