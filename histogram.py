import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import cosine
import seaborn as sns

# Funções auxiliares
def smiles_to_fingerprint(smiles, radius=2):
    """Converte SMILES para fingerprint de Morgan."""
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius) if mol else None
