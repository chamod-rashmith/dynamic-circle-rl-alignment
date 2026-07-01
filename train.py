import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import copy

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=200, lr=0.01, batch_size=64, patience=25, min_delta=0.0001):
    """
    Trains the given PyTorch model using Binary Cross Entropy Loss and Adam optimizer.
    Includes Early Stopping to prevent overfitting.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create DataLoader for batching
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Early Stopping variables
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    
    print(f"Starting training for {epochs} epochs (with Early Stopping patience={patience})...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(X_train)
        
        # Validation checks for early stopping
        val_loss = None
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_preds_prob = model(X_val)
                val_loss = criterion(val_preds_prob, y_val).item()
                
                # Check for improvement
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
        
        # Evaluate & Print metrics periodically
        if epoch % 20 == 0 or epoch == 1 or (val_loss is not None and patience_counter >= patience):
            model.eval()
            with torch.no_grad():
                train_preds = (model(X_train) >= 0.5).float()
                train_acc = (train_preds == y_train).float().mean().item() * 100
                
                val_info = ""
                if X_val is not None and y_val is not None:
                    val_preds = (model(X_val) >= 0.5).float()
                    val_acc = (val_preds == y_val).float().mean().item() * 100
                    val_info = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%{val_info}")
        
        # Early stopping trigger
        if X_val is not None and y_val is not None and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}! Best Val Loss was: {best_val_loss:.4f}")
            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)
            break
                
    return model
