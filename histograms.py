import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import cosine
import seaborn as sns

# Funções auxiliares
def smiles_to_fingerprint(smiles, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius) if mol else None

def process_batch(fingerprints, calculate_function):
    results = []
    for fp in fingerprints:
        results.extend(calculate_function(fp, fingerprints))
    return results

def calculate_similarity(fingerprint, fingerprints, similarity_metric):
    similarities = []
    for fp in fingerprints:
        if similarity_metric == 'cosine':
            arr1, arr2 = np.zeros((1,)), np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, arr1)
            DataStructs.ConvertToNumpyArray(fp, arr2)
            similarity = 1 - cosine(arr1, arr2)
        elif similarity_metric == 'tanimoto':
            similarity = DataStructs.TanimotoSimilarity(fingerprint, fp)
        elif similarity_metric == 'dice':
            similarity = DataStructs.DiceSimilarity(fingerprint, fp)
        similarities.append(similarity)
    return similarities

def calculate_distance(fingerprint, fingerprints, distance_metric):
    distances = []
    for fp in fingerprints:
        if distance_metric == 'hamming':
            dist = sum(1 for a, b in zip(fingerprint, fp) if a != b)
        elif distance_metric == 'manhattan':
            arr1, arr2 = np.zeros((1,)), np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, arr1)
            DataStructs.ConvertToNumpyArray(fp, arr2)
            dist = np.sum(np.abs(arr1 - arr2))
        distances.append(dist)
    return distances

def process_in_batches(data, batch_size, process_function, metric):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        results.extend(process_batch(batch, lambda fp, fps: process_function(fp, fps, metric)))
    return results

# Carregar o arquivo .tsv
file_path = './chembl_cluster_hits.tsv'  # Substitua pelo caminho correto do arquivo
data = pd.read_csv(file_path, sep='\t')

# Preparar fingerprints
data['fingerprints'] = data['canonical_smiles'].apply(smiles_to_fingerprint)
valid_fps = data['fingerprints'].dropna().tolist()

# Definir o tamanho do lote
batch_size = 1024  # Ajuste conforme a capacidade da sua máquina

# Calcular todas as similaridades e distâncias por bateladas
similarity_metrics = ['tanimoto', 'dice', 'cosine']
distance_metrics = ['hamming', 'manhattan']
all_similarities = {metric: process_in_batches(valid_fps, batch_size, calculate_similarity, metric) for metric in similarity_metrics}
all_distances = {metric: process_in_batches(valid_fps, batch_size, calculate_distance, metric) for metric in distance_metrics}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import cosine
import seaborn as sns

# Funções auxiliares
def smiles_to_fingerprint(smiles, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius) if mol else None

def process_batch(fingerprints, calculate_function):
    results = []
    for fp in fingerprints:
        results.extend(calculate_function(fp, fingerprints))
    return results

def calculate_similarity(fingerprint, fingerprints, similarity_metric):
    similarities = []
    for fp in fingerprints:
        if similarity_metric == 'cosine':
            arr1, arr2 = np.zeros((1,)), np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, arr1)
            DataStructs.ConvertToNumpyArray(fp, arr2)
            similarity = 1 - cosine(arr1, arr2)
        elif similarity_metric == 'tanimoto':
            similarity = DataStructs.TanimotoSimilarity(fingerprint, fp)
        elif similarity_metric == 'dice':
            similarity = DataStructs.DiceSimilarity(fingerprint, fp)
        similarities.append(similarity)
    return similarities

def calculate_distance(fingerprint, fingerprints, distance_metric):
    distances = []
    for fp in fingerprints:
        if distance_metric == 'hamming':
            dist = sum(1 for a, b in zip(fingerprint, fp) if a != b)
        elif distance_metric == 'manhattan':
            arr1, arr2 = np.zeros((1,)), np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, arr1)
            DataStructs.ConvertToNumpyArray(fp, arr2)
            dist = np.sum(np.abs(arr1 - arr2))
        distances.append(dist)
    return distances

def process_in_batches(data, batch_size, process_function, metric):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        results.extend(process_batch(batch, lambda fp, fps: process_function(fp, fps, metric)))
    return results

# Carregar o arquivo .tsv
file_path = './chembl_cluster_hits.tsv'  # Substitua pelo caminho correto do arquivo
data = pd.read_csv(file_path, sep='\t')

# Preparar fingerprints
data['fingerprints'] = data['canonical_smiles'].apply(smiles_to_fingerprint)
valid_fps = data['fingerprints'].dropna().tolist()

# Definir o tamanho do lote
batch_size = 1024  # Ajuste conforme a capacidade da sua máquina

# Calcular todas as similaridades e distâncias por bateladas
similarity_metrics = ['tanimoto', 'dice', 'cosine']
distance_metrics = ['hamming', 'manhattan']
all_similarities = {metric: process_in_batches(valid_fps, batch_size, calculate_similarity, metric) for metric in similarity_metrics}
all_distances = {metric: process_in_batches(valid_fps, batch_size, calculate_distance, metric) for metric in distance_metrics}

# Inicializar figura para os subplots
fig, axs = plt.subplots(3, 2, figsize=(13, 13))  # 3 linhas, 2 colunas

# Plotar histogramas das similaridades
for ax, (metric, values) in zip(axs[:, 0], all_similarities.items()):
    ax.hist(values, bins=50, color='skyblue', edgecolor='black')
    ax.set_title(f'Similarity - {metric.capitalize()}', fontsize=10)
    #ax.set_xlabel('Similaridade')
    ax.set_ylabel('Frequency')

# Plotar histogramas das distâncias
for ax, (metric, values) in zip(axs[:, 1], all_distances.items()):
    ax.hist(values, bins=50, color='salmon', edgecolor='black')
    ax.set_title(f'Distance - {metric.capitalize()}', fontsize=10)
    #ax.set_xlabel('Distância')
    ax.set_ylabel('Frequency')

# Plotar histograma dos valores de pChEMBL no último subplot
axs[2, 1].hist(data['pchembl_value'].dropna(), bins=20, color='green', edgecolor='black')
axs[2, 1].set_title('ChEMBL value', fontsize=10)
#axs[2, 1].set_xlabel('pChEMBL Value')
axs[2, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('histogram_similarity_distance.png')
plt.show()

