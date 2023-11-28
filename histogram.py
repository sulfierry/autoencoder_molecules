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


def calculate_similarity(fingerprints, similarity_metric):
    """Calcula similaridades entre todos os pares de fingerprints."""
    similarities = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            if similarity_metric == 'cosine':
                arr1, arr2 = np.zeros((1,)), np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fingerprints[i], arr1)
                DataStructs.ConvertToNumpyArray(fingerprints[j], arr2)
                similarity = 1 - cosine(arr1, arr2)
            elif similarity_metric == 'tanimoto':
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            elif similarity_metric == 'dice':
                similarity = DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(similarity)
    return similarities

def calculate_distance(fingerprints, distance_metric):
    """Calcula distâncias entre todos os pares de fingerprints."""
    distances = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            if distance_metric == 'hamming':
                dist = sum(1 for a, b in zip(fingerprints[i], fingerprints[j]) if a != b)
            elif distance_metric == 'manhattan':
                arr1, arr2 = np.zeros((1,)), np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fingerprints[i], arr1)
                DataStructs.ConvertToNumpyArray(fingerprints[j], arr2)
                dist = np.sum(np.abs(arr1 - arr2))
            distances.append(dist)
    return distances


# Carregar o arquivo .tsv
file_path = './kinases_grupo4.tsv'  # Substitua pelo caminho correto do arquivo
data = pd.read_csv(file_path, sep='\t')

# Preparar fingerprints
data['fingerprints'] = data['canonical_smiles'].apply(smiles_to_fingerprint)
valid_fps = data['fingerprints'].dropna().tolist()

# Calcular todas as similaridades e distâncias
similarity_metrics = ['tanimoto', 'dice', 'cosine']
distance_metrics = ['hamming', 'manhattan']
all_similarities = {metric: calculate_similarity(valid_fps, metric) for metric in similarity_metrics}
all_distances = {metric: calculate_distance(valid_fps, metric) for metric in distance_metrics}

# Inicializar figura para os subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # 3 linhas, 2 colunas


# Plotar histogramas das similaridades
for ax, (metric, values) in zip(axs[:, 0], all_similarities.items()):
    ax.hist(values, bins=50, color='skyblue', edgecolor='black')
    ax.set_title(f'Histograma de Similaridade - {metric.capitalize()}')
    #ax.set_xlabel('Similaridade')
    ax.set_ylabel('Frequência')

# Plotar histogramas das distâncias
for ax, (metric, values) in zip(axs[:, 1], all_distances.items()):
    ax.hist(values, bins=50, color='salmon', edgecolor='black')
    ax.set_title(f'Histograma de Distância - {metric.capitalize()}')
    #ax.set_xlabel('Distância')
    ax.set_ylabel('Frequência')
