"""
Simple inference script for the trained MPNN model
"""
import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_mpnn import MPNNMLX
from utils import Smi2GraphMLX

def load_model(model_path='checkpoints/best_model.npz'):
    """Load a trained model from disk"""
    model = MPNNMLX()
    # Load the flattened parameters
    flat_params = mx.load(model_path)
    # Unflatten back to the nested structure
    weights = tree_unflatten(list(flat_params.items()))
    model.update(weights)
    model.eval()
    return model

def predict_bbbp(model, smiles):
    """
    Predict blood-brain barrier permeability for a molecule
    
    Args:
        model: Trained MPNN model
        smiles: SMILES string of the molecule
    
    Returns:
        float: Predicted probability of BBB permeability (0-1)
    """
    graph = Smi2GraphMLX(smiles, label=None)
    if graph is None:
        print(f"Failed to parse SMILES: {smiles}")
        return None
    
    # Make prediction
    pred = model(graph)
    return float(pred.item())

if __name__ == "__main__":
    # Load model
    print("Loading model from checkpoints/best_model.npz...")
    model = load_model()
    print("✓ Model loaded!\n")
    
    # Example molecules
    test_molecules = [
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(O)=O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Dopamine", "NCCc1ccc(O)c(O)c1"),
    ]
    
    print("Testing predictions:")
    print("-" * 60)
    for name, smiles in test_molecules:
        pred = predict_bbbp(model, smiles)
        if pred is not None:
            result = "Permeable" if pred > 0.5 else "Non-permeable"
            print(f"{name:15s} | Score: {pred:.4f} | {result}")
    print("-" * 60)

