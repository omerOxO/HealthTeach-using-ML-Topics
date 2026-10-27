"""RDKit helpers: SMILES standardisation, InChIKeys and Morgan fingerprints."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from .config import FP_BITS, FP_RADIUS

RDLogger.DisableLog("rdApp.*")

_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def standard_mol(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        return _uncharger.uncharge(_chooser.choose(mol))
    except Exception:
        return mol


def inchikey_block(smiles):
    """First InChIKey block (connectivity only), robust to salt, charge and stereo differences."""
    mol = standard_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def morgan(smiles_list):
    out = np.zeros((len(smiles_list), FP_BITS), dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        mol = standard_mol(s)
        if mol is not None:
            out[i] = _fpgen.GetFingerprintAsNumPy(mol)
    return out
