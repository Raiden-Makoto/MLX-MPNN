"""
Simple inference script for the trained MPNN model
"""
import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_mpnn import MPNNMLX
from utils import Smi2GraphMLX
import pandas as pd

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
    
    # Load QM9 molecules
    print("Loading QM9 molecules...")
    df = pd.read_csv("QM9.csv", usecols=["smiles"]).sample(6767)
    print(f"Processing {len(df)} molecules...\n")
    
    # Predict BBBP for each molecule
    predictions = []
    for idx, smiles in enumerate(df['smiles']):
        pred = predict_bbbp(model, smiles)
        predictions.append(pred if pred is not None else 0.0)
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(df)} molecules...")
    
    df['p_np'] = predictions
    df['prediction'] = df['p_np'].apply(lambda x: "Permeable" if x > 0.5 else "Non-permeable")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Prediction Summary:")
    print("=" * 60)
    print(f"Total molecules: {len(df)}")
    print(f"Permeable: {(df['p_np'] > 0.5).sum()} ({(df['p_np'] > 0.5).sum() / len(df) * 100:.1f}%)")
    print(f"Non-permeable: {(df['p_np'] <= 0.5).sum()} ({(df['p_np'] <= 0.5).sum() / len(df) * 100:.1f}%)")
    print(f"Mean score: {df['p_np'].mean():.4f}")
    print("=" * 60)
    
    # Show first few predictions
    print("\nFirst 10 predictions:")
    print("-" * 60)
    for idx, row in df.head(10).iterrows():
        print(f"{row['smiles'][:40]:40s} | Score: {row['p_np']:.4f} | {row['prediction']}")
    print("-" * 60)
    
    # Save results
    output_file = "qm9_mlx.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")

