import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from mlx_graphs.loaders import Dataloader # type: ignore
from utils import Smi2GraphMLX
from mlx_mpnn import MPNNMLX

from tqdm import tqdm
import numpy as np
import os
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

df = pd.read_csv("BBBP.csv", usecols=["smiles", "p_np"])
graphs = []

for _, row in df.iterrows():
    smiles = row["smiles"]
    label = row["p_np"]
    graph = Smi2GraphMLX(smiles, label=label)
    if graph is not None:  # Skip failed molecules
        graphs.append(graph)

print("Processed", len(graphs), "graphs.")

train_graphs, other_graphs = train_test_split(graphs, test_size=0.2, random_state=67)
val_graphs, test_graphs = train_test_split(other_graphs, test_size=0.5, random_state=67)

print("--------------------------------")
print("Train dataset size:", len(train_graphs))
print("Validation dataset size:", len(val_graphs))
print("Test dataset size:", len(test_graphs))
print("--------------------------------")

train_loader = Dataloader(train_graphs, batch_size=32, shuffle=True)
val_loader = Dataloader(val_graphs, batch_size=32, shuffle=False)
test_loader = Dataloader(test_graphs, batch_size=32, shuffle=False)

model = MPNNMLX() # stick to the defaults

# Create checkpoints directory
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Checkpoints will be saved to: {checkpoint_dir}/")

# Hyperparameters
num_epochs = 40
initial_lr = 5e-4
min_lr = 1e-6
patience = 15
warmup_epochs = 5
checkpoint_interval = 10  # Save checkpoint every N epochs

# Optimizer with cosine learning rate schedule
def get_cosine_schedule(epoch, warmup_epochs, num_epochs, initial_lr, min_lr):
    """Cosine annealing schedule with warmup"""
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

def save_checkpoint(model, epoch, val_loss, train_loss, checkpoint_path):
    """Save model checkpoint with metadata"""
    flat_params = tree_flatten(model.parameters())
    
    # Save model parameters
    mx.savez(checkpoint_path, **dict(flat_params))
    
    # Save metadata as a separate JSON file
    metadata = {
        'epoch': epoch,
        'val_loss': float(val_loss),
        'train_loss': float(train_loss),
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    metadata_path = checkpoint_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

optimizer = optim.Adam(learning_rate=initial_lr)

# Loss and metric functions
def compute_loss(batch):
    preds = model(batch)
    return nn.losses.mse_loss(preds, batch.y)

def metric_fn(pred, target):
    return mx.mean(mx.abs(pred - target))

# Create the value-and-grad function
loss_and_grad = nn.value_and_grad(model, compute_loss)

# Early stopping variables
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_weights = None

print(f"\nTraining with:")
print(f"  Initial LR: {initial_lr}")
print(f"  Min LR: {min_lr}")
print(f"  Warmup epochs: {warmup_epochs}")
print(f"  Early stopping patience: {patience}")
print("--------------------------------\n")

# Training loop
for epoch in range(1, num_epochs + 1):
    # Update learning rate
    current_lr = get_cosine_schedule(epoch - 1, warmup_epochs, num_epochs, initial_lr, min_lr)
    optimizer.learning_rate = current_lr
    
    # --- Training ---
    model.train()
    train_loss = 0.0
    train_batches = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        loss, grads = loss_and_grad(batch)
        train_loss += loss.item()
        # Apply gradients and update model parameters
        model.update(optimizer.apply_gradients(grads, model.parameters()))
        mx.eval(model.parameters())
        train_batches += 1
    avg_train_loss = train_loss / train_batches

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_metric = 0.0
    val_batches = 0
    for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
        preds = model(batch)
        val_loss   += nn.losses.mse_loss(preds, batch.y).item()
        val_metric += metric_fn(preds, batch.y).item()
        val_batches += 1
    avg_val_loss   = val_loss   / val_batches
    avg_val_metric = val_metric / val_batches

    # Print metrics
    print(
        f"Epoch {epoch:02d}  "
        f"LR: {current_lr:.2e}  "
        f"Train Loss: {avg_train_loss:.4f}  "
        f"Val Loss: {avg_val_loss:.4f}  "
        f"Val MAE: {avg_val_metric:.4f}"
    )
    
    # Early stopping and model checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        # Save best model weights
        best_model_weights = {k: v.copy() for k, v in model.parameters().items()}
        # Save best model to disk
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.npz')
        save_checkpoint(model, epoch, avg_val_loss, avg_train_loss, best_checkpoint_path)
        print(f"  ✓ New best model! (Val Loss: {best_val_loss:.4f}) - Saved to {best_checkpoint_path}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            print(f"  Best validation loss: {best_val_loss:.4f}")
            break
    
    # Regular checkpoint every N epochs
    if epoch % checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.npz')
        save_checkpoint(model, epoch, avg_val_loss, avg_train_loss, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    print()

# Restore best model weights for final evaluation
if best_model_weights is not None:
    print("\n✓ Restoring best model weights for final evaluation...")
    model.update(best_model_weights)
    print(f"   Best checkpoint: {os.path.join(checkpoint_dir, 'best_model.npz')}")
    print(f"   Best validation loss: {best_val_loss:.4f}")

# --- Final Test Evaluation ---
print("\n" + "="*50)
print("Evaluating on test set...")
print("="*50)
model.eval()
test_loss = 0.0
test_metric = 0.0
test_batches = 0
for batch in tqdm(test_loader, desc="Testing"):
    preds = model(batch)
    test_loss += nn.losses.mse_loss(preds, batch.y).item()
    test_metric += metric_fn(preds, batch.y).item()
    test_batches += 1

avg_test_loss = test_loss / test_batches
avg_test_metric = test_metric / test_batches

print(f"\nFinal Test Results:")
print(f"  Test Loss: {avg_test_loss:.4f}")
print(f"  Test MAE:  {avg_test_metric:.4f}")
print("="*50)