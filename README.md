# Track_Me: 2D Dynamic Circle Classification & Reinforcement Learning Alignment

An end-to-end Deep Learning project demonstrating **Supervised Pre-training (SFT)**, **Reinforcement Learning (RL) Fine-Tuning (Alignment)**, and **Explainable AI (XAI)** to solve a dynamic geometric classification problem.

---

## 📌 Project Overview

This project trains a PyTorch Multi-Layer Perceptron (MLP) to determine whether a 2D point $(x, y)$ lies **inside** or **outside** a circle of dynamic radius $r$.

The classification boundary follows the mathematical relationship:
$$x^2 + y^2 \le r^2$$

### Key Highlights
1. **Dynamic Features**: Instead of classifying points inside a fixed boundary, the model accepts `[x, y, r]` as inputs, learning a parameterized decision boundary.
2. **Hybrid Training**:
   * **Supervised Learning**: Model is trained on coordinates in $[-3.0, 3.0]$ with Early Stopping, achieving **~98.5% Accuracy**.
   * **Reinforcement Learning (REINFORCE)**: The pre-trained model is aligned using Policy Gradients (REINFORCE) over an expanded domain of $[-15.0, 15.0]$ with custom rewards.
3. **Interpretability & Diagnostics (XAI)**:
   * Feature attribution calculated using **Integrated Gradients (IG)**.
   * Model statistics and weight drift visual comparisons (Supervised vs. RL).

---

## 🏗️ Model Architecture

The [BaseModel](model/base_model.py) is a feedforward neural network built with PyTorch:

```python
self.model = nn.Sequential(
    nn.Linear(in_features=3, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=1),
    nn.Sigmoid()
)
```

---

## 📂 Project Structure

```bash
Track_Me/
│
├── model/
│   └── base_model.py       # PyTorch MLP Model definition
│
├── evaluate/
│   ├── evaluate.py         # Simple point evaluation script
│   ├── plot_stats.py       # Weights/biases stats and distributions plotter
│   ├── compare_weights.py  # Supervised vs. RL weight drift analyzer
│   ├── xai_explain.py      # Integrated Gradients feature attribution
│   └── plots/              # Saved visualizations
│       ├── model_weight_stats.png
│       ├── multi_radius_decision_boundary.png
│       ├── rl_training_rewards.png
│       ├── weight_comparison.png
│       └── xai_explanation.png
│
├── rl/
│   └── rl_finetune.py      # RL Fine-tuning using REINFORCE
│
├── output/
│   ├── dynamic_radius_model.pth  # Supervised trained weights
│   └── rl_finetuned_model.pth    # RL aligned weights
│
├── main.py                 # Supervised training orchestration script
├── train.py                # Reusable supervised training loop (with Early Stopping)
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install torch numpy matplotlib
```

### 2. Run Supervised Training
Train the model on the $[-3.0, 3.0]$ range. This saves the model to `output/` and generates decision boundary plots:
```bash
python main.py
```

### 3. Run Reinforcement Learning Fine-tuning
Align the pre-trained supervised model over a larger domain $[-15.0, 15.0]$ using custom rewards:
```bash
python rl/rl_finetune.py
```


### 4. Explain Model Decisions (XAI)
Run the Integrated Gradients feature attribution script to inspect how individual features (coordinates vs. radius) impact predictions:
```bash
python evaluate/xai_explain.py
```

---

## 📊 Performance & Visualizations

All generated plots are saved inside the [evaluate/plots/](evaluate/plots) directory:

### 1. Dynamic Decision Boundary
Plots the decision boundary (green) against the true mathematical circle (dashed line) for multiple input radiuses ($r = 0.5, 1.5, 2.5$):
* File: `evaluate/plots/multi_radius_decision_boundary.png`

### 2. Reinforcement Learning Rewards
Visualizes the training rewards over RL epochs. It starts at `+0.98` thanks to the supervised warm start, demonstrating stable convergence without catastrophic forgetting:
* File: `evaluate/plots/rl_training_rewards.png`

### 3. Integrated Gradients Attribution
Shows how the model assigns positive attribution to the Radius (supports INSIDE) and negative attribution to the coordinates (supports OUTSIDE):
* File: `evaluate/plots/xai_explanation.png`
