from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
import numpy as np

# Hide noisy RDKit parse logs.
RDLogger.DisableLog("rdApp.*")


def clean_smiles(smiles: str) -> str:
    """Clean noisy SMILES strings from compound metadata."""
    if not isinstance(smiles, str):
        return ""

    smiles = smiles.strip()

    # Remove quotes if present.
    smiles = smiles.replace('"', "").replace("'", "")

    # Some rows have trailing commas, e.g. "CCO,"
    # Split is more aggressive than rstrip.
    if "," in smiles:
        smiles = smiles.split(",")[0]

    # Some rows may contain multiple chunks separated by whitespace.
    smiles = smiles.split()[0] if smiles else ""

    return smiles.strip()


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert a SMILES string into a Morgan fingerprint."""
    smiles = clean_smiles(smiles)

    if smiles == "":
        return np.zeros(n_bits, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr
