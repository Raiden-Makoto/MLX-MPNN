import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from mlx_graphs.loaders import Dataloader # type: ignore
from utils import Smi2GraphMLX
from mlx_mpnn import MPNNMLX

from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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
optimizer = optim.Adam(learning_rate=5e-4)

# Loss and metric functions
def compute_loss(batch):
    preds = model(batch)
    return nn.losses.mse_loss(preds, batch.y)

def metric_fn(pred, target):
    return mx.mean(mx.abs(pred - target))

# Create the value-and-grad function
loss_and_grad = nn.value_and_grad(model, compute_loss)

# Training loop
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        loss, grads = loss_and_grad(batch)
        train_loss += loss.item()
        # Apply gradients to update parameters
        optimizer.apply_gradients(grads, model.parameters())
        mx.eval(model.parameters())
    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_metric = 0.0
    for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
        preds = model(batch)
        val_loss   += nn.losses.mse_loss(preds, batch.y).item()
        val_metric += metric_fn(preds, batch.y).item()
    avg_val_loss   = val_loss   / len(val_loader)
    avg_val_metric = val_metric / len(val_loader)

    print(
        f"Epoch {epoch:02d}  "
        f"Train Loss: {avg_train_loss:.4f}  "
        f"Val Loss:   {avg_val_loss:.4f}  "
        f"Val MAE:    {avg_val_metric:.4f}"
    )