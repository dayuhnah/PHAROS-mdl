from rdkit import Chem
import torch
from torch_geometric.data import Data

from src.data.featurize_smiles import clean_smiles


ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "H", "Unknown"
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices):
    if value not in choices:
        value = choices[-1]
    return [1.0 if value == choice else 0.0 for choice in choices]


def atom_features(atom):
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES:
        symbol = "Unknown"

    features = []

    features += one_hot(symbol, ATOM_TYPES)
    features.append(float(atom.GetDegree()))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetTotalNumHs()))
    features.append(float(atom.GetImplicitValence()))
    features.append(float(atom.GetIsAromatic()))

    return features


def bond_features(bond):
    bond_type = bond.GetBondType()

    features = []

    features += [1.0 if bond_type == bt else 0.0 for bt in BOND_TYPES]
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))

    return features


def smiles_to_graph(smiles: str) -> Data | None:
    smiles = clean_smiles(smiles)

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    atom_feature_list = [atom_features(atom) for atom in mol.GetAtoms()]

    if len(atom_feature_list) == 0:
        return None

    x = torch.tensor(atom_feature_list, dtype=torch.float)

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        features = bond_features(bond)

        edge_indices.append([i, j])
        edge_indices.append([j, i])

        edge_attrs.append(features)
        edge_attrs.append(features)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_TYPES) + 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)