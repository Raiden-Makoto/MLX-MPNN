from rdkit import Chem # type: ignore
import mlx.core as mx # type: ignore
from mlx_graphs.data import GraphData # type: ignore

def Smi2GraphMLX(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Failed to parse SMILES:", smiles)
        return None

    mol = Chem.AddHs(mol)
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
        ]
        node_features.append(features)
    node_features = mx.array(node_features, dtype=mx.float32)

    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        bond_type = float(bond.GetBondTypeAsDouble())
        edge_attr = [
            bond_type,
            float(bond.GetIsAromatic()),
            float(bond.GetIsConjugated()),
        ]
        edge_features.extend([edge_attr, edge_attr])
    edge_index = mx.array(edge_index, dtype=mx.int64).T
    edge_features = mx.array(edge_features, dtype=mx.float32)

    batch = mx.zeros(node_features.shape[0], dtype=mx.int64)  # single graph in batch

    graph = GraphData(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        batch=batch,
    )
    return graph
