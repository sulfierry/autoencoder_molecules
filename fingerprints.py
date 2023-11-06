from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

# Função para processar um chunk de dados
def process_chunk(chunk, threshold):
    # Converter SMILES em objetos de molécula RDKit e calcular fingerprints
    chembl_molecules = (Chem.MolFromSmiles(smile) for smile in chunk['chembl_smile'])
    pkidb_molecules = (Chem.MolFromSmiles(smile) for smile in chunk['pkidb_smile'])

    chembl_fps = (AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in chembl_molecules if mol)
    pkidb_fps = (AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in pkidb_molecules if mol)

    # Calcular as similaridades de Tanimoto
    tanimoto_similarities = [DataStructs.FingerprintSimilarity(chembl_fp, pkidb_fp)
                             for chembl_fp, pkidb_fp in zip(chembl_fps, pkidb_fps)]

    # Adicionar as similaridades de Tanimoto ao chunk DataFrame
    chunk['tanimoto_similarity'] = tanimoto_similarities

    # Filtrar com base no limiar de similaridade de Tanimoto
    filtered_chunk = chunk[chunk['tanimoto_similarity'] >= threshold]

    return chunk, filtered_chunk

# Função principal
def main():
    file_path = './similar_molecules_3.tsv'
    threshold = 0.6
    chunksize = 10000  # Ajuste este valor de acordo com a sua memória disponível
