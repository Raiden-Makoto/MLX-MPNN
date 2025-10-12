from rdkit import Chem # type: ignore
import mlx.core as mx # type: ignore
from mlx_graphs.data import GraphData # type: ignore

def Smi2GraphMLX(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Failed to parse SMILES:", smiles)
        return None

    mol = Chem.AddHs(mol)
    
    # Create 12-dimensional node features (matching the notebook format)
    node_features = []
    for atom in mol.GetAtoms():
        element = atom.GetSymbol()
        degree = atom.GetDegree()
        valence = atom.GetTotalValence()
        num_h = atom.GetTotalNumHs()
        
        # One-hot encoding for common elements (9 dimensions)
        feat = [
            float(element == symbol) 
            for symbol in ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        ] + [degree, valence, num_h]
        
        node_features.append(feat)
    node_features = mx.array(node_features, dtype=mx.float32)

    # Create 4-dimensional edge features (matching the notebook format)
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        
        bt = bond.GetBondType()
        edge_attr = [
            float(bt == Chem.rdchem.BondType.SINGLE),
            float(bt == Chem.rdchem.BondType.DOUBLE),
            float(bt == Chem.rdchem.BondType.TRIPLE),
            float(bond.GetIsConjugated())
        ]
        edge_features.extend([edge_attr, edge_attr])
        
    edge_index = mx.array(edge_index, dtype=mx.int32).T
    edge_features = mx.array(edge_features, dtype=mx.float32)

    # Create graph data with optional label
    # Note: Don't set 'batch' here - the Dataloader will handle it during batching
    graph_dict = {
        'x': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_features,
    }
    
    if label is not None:
        graph_dict['y'] = mx.array([label], dtype=mx.float32)
    
    graph = GraphData(**graph_dict)
    return graph
