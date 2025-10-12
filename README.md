# MLX-MPNN

Message Passing Neural Network for blood-brain barrier permeability prediction using MLX and MLX Graphs for Apple Silicon.

## Overview

This project implements a Graph Neural Network (GNN) using Message Passing Neural Networks (MPNN) to predict whether molecules can cross the blood-brain barrier (BBB). Built specifically for Apple Silicon using the MLX framework, it provides efficient training and inference on M1/M2/M3 Macs.

## Features

- 🧠 **MPNN Architecture**: Edge-conditioned message passing with GRU state updates
- 🚀 **Apple Silicon Optimized**: Uses MLX for hardware-accelerated training on M-series chips
- 📊 **Smart Training**: Cosine learning rate scheduling with warmup and early stopping
- 💾 **Checkpoint Management**: Automatic best model saving + regular checkpoints every 10 epochs
- 🔬 **Easy Inference**: Simple API for predicting BBB permeability from SMILES strings

## Installation

### Requirements
- Python 3.11+
- Apple Silicon Mac 

### Setup

```bash
# Clone the repository
git clone https://github.com/Raiden-Makoto/MLX-MPNN.git
cd MLX-MPNN

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install mlx mlx-graphs pandas rdkit numpy jupyter tqdm matplotlib scikit-learn
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

## Project Structure

```
MLX-MPNN/
├── train.py              # Training script with checkpointing
├── predict.py            # Inference script for trained models
├── mlx_mpnn.py          # MPNN model implementation
├── EdgeNetwork.py       # Edge-conditioned message passing layer
├── utils.py             # Data processing utilities (SMILES → Graph)
├── list_checkpoints.py  # View saved checkpoints
├── BBBP.csv            # Blood-Brain Barrier Permeability dataset
├── checkpoints/        # Model checkpoints (created during training)
└── README.md           # This file
```

## Usage

### Training

Train the model on the BBBP dataset:

```bash
python train.py
```

**Training Features:**
- **Cosine LR Schedule**: Smooth learning rate decay with 5-epoch warmup
- **Early Stopping**: Stops after 15 epochs without improvement
- **Auto Checkpointing**: 
  - Best model saved to `checkpoints/best_model.npz`
  - Regular checkpoints every 10 epochs: `checkpoints/epoch_XXX.npz`
- **Progress Tracking**: Real-time loss and metrics with tqdm progress bars

**Hyperparameters** (in `train.py`):
```python
num_epochs = 100
initial_lr = 5e-4
min_lr = 1e-6
patience = 15
warmup_epochs = 5
checkpoint_interval = 10
batch_size = 32
```

### Inference

Use a trained model to predict BBB permeability:

```python
from predict import load_model, predict_bbbp

# Load the best model
model = load_model('checkpoints/best_model.npz')

# Predict for a molecule (SMILES string)
smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
score = predict_bbbp(model, smiles)

print(f"BBB Permeability Score: {score:.4f}")
print(f"Prediction: {'Permeable' if score > 0.5 else 'Non-permeable'}")
```

Or run the example script:
```bash
python predict.py
```

### Managing Checkpoints

List all available checkpoints with metadata:

```bash
python list_checkpoints.py
```

Output example:
```
================================================================================
Available Checkpoints in checkpoints/
================================================================================

📦 best_model.npz
   Epoch:      23
   Train Loss: 0.0842
   Val Loss:   0.0891
   Saved:      2025-10-12T19:30:45.123456

📦 epoch_010.npz
   Epoch:      10
   Train Loss: 0.1234
   Val Loss:   0.1156
   Saved:      2025-10-12T19:15:22.654321

================================================================================
Total: 2 checkpoint(s)
================================================================================
```

## Model Architecture

### MPNN Details

- **Node Features** (12-dim per atom):
  - 9D: One-hot element encoding (C, N, O, F, P, S, Cl, Br, I)
  - 1D: Atom degree
  - 1D: Total valence
  - 1D: Number of hydrogens

- **Edge Features** (4-dim per bond):
  - 1D: Single bond indicator
  - 1D: Double bond indicator
  - 1D: Triple bond indicator
  - 1D: Conjugation status

- **Architecture**:
  ```
  Input (SMILES) → Molecular Graph
       ↓
  Node Linear: 12 → 32
       ↓
  4× Message Passing Steps:
       Edge Network (edge features → weight matrix)
       Message Passing (weighted neighbor aggregation)
       GRU Cell (state update)
       ↓
  Global Mean Pooling
       ↓
  Readout MLP: 32 → 256 → 1
       ↓
  Sigmoid → BBB Permeability Score [0, 1]
  ```

### Key Components

1. **EdgeNetwork**: Learns edge-specific weight matrices from bond features
2. **GRU Cell**: Manual implementation with reset/update/candidate gates
3. **Global Pooling**: Aggregates node features to graph-level representation

## Dataset

The model is trained on the **BBBP (Blood-Brain Barrier Penetration)** dataset:
- ~2,000 molecules with BBB permeability labels
- Binary classification: Permeable (1) vs Non-permeable (0)
- Data from: [MoleculeNet benchmark](https://moleculenet.org/)

## Performance

Expected results (will vary based on random seed):
- **Validation Loss**: ~0.08-0.10
- **Test Loss**: ~0.09-0.11
- **Test MAE**: ~0.20-0.25

## Technical Notes

### MLX-Graphs Bug Workaround

The code includes a workaround for an mlx-graphs v0.0.8 bug where `batch_indices` can be 1-3 elements shorter than expected. The model automatically pads these indices during forward passes.

### Serialization

Models are saved using MLX's native `.npz` format:
- Parameters are flattened with `mlx.utils.tree_flatten`
- Metadata (epoch, losses, timestamp) saved as JSON
- Loading uses `mlx.utils.tree_unflatten` to restore structure

## Development

### Running Tests

The repository includes test scripts (not tracked in git) for development:
- `test_edge_network.py`: Test message passing layer
- `test_training.py`: Verify gradient updates
- `test_save_load.py`: Check model serialization

## Citation

If you use this code, please cite:
- [MLX Framework](https://github.com/ml-explore/mlx)
- [MLX-Graphs](https://github.com/mlx-graphs/mlx-graphs)
- Original MPNN paper: Gilmer et al. "Neural Message Passing for Quantum Chemistry" (2017)

## License

MIT License
