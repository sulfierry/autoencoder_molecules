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

# Função para agrupar por 'pchembl_value'
def pchembl_group(value):
    if pd.isna(value):
        return 'sem_pchembl'
    elif 1 < value < 4:
        return 'grupo1'
    elif 4 <= value < 6:
        return 'grupo2'
    elif 6 <= value < 8:
        return 'grupo3'
    elif 8 <= value <= 10:
        return 'grupo4'
    else:
        return 'outros'

# Aplicar agrupamento
data['pchembl_group'] = data['pchembl_value'].apply(pchembl_group)


# Filtrar e salvar kinases sem 'pchembl_value'
kinases_sem_pchembl = data[data['pchembl_group'] == 'sem_pchembl']
kinases_sem_pchembl.to_csv('./kinases_sem_pchembl_value.tsv', sep='\t', index=False)

# Remover kinases sem 'pchembl_value' para análise t-SNE
data = data[data['pchembl_group'] != 'sem_pchembl']

# Preparar dados para t-SNE e plotagem
tsne_results = []
group_labels = []
