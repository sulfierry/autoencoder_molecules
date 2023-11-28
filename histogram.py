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
