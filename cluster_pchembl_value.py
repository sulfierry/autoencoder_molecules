import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Carregar dados
file_path = './kinase_ligands_pchembl_Value.tsv'
data = pd.read_csv(file_path, sep='\t')


# Função para converter SMILES em fingerprint
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) if mol else None
