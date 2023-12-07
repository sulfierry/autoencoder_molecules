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

