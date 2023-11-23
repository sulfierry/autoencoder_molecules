import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm

def smiles_to_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)
    except:
        return None
    return None

def process_chunk(chunk, pkidb_fingerprints):
    non_matching_smiles = []
    for _, row in chunk.iterrows():
        chembl_fp = smiles_to_fingerprint(row['canonical_smiles'])
        if chembl_fp is not None and not any(DataStructs.FingerprintSimilarity(chembl_fp, fp) == 1.0 for fp in pkidb_fingerprints if fp is not None):
            non_matching_smiles.append(row['canonical_smiles'])
    return non_matching_smiles

# Carregar os SMILES do pkidb
pkidb_smiles = pd.read_csv('./pkidb_2023-06-30.tsv', sep='\t')
pkidb_smiles['fingerprint'] = pkidb_smiles['Canonical_Smiles'].apply(smiles_to_fingerprint)

# Preparar a lista de fingerprints do pkidb para comparação
pkidb_fingerprints = [fp for fp in pkidb_smiles['fingerprint'].tolist() if fp is not None]

# Utilizar todas as CPUs disponíveis
cpu_count = os.cpu_count()

# Processar o arquivo ChEMBL em partes
chunksize = 10000
non_matching_results = []

with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    futures = []
    for chunk in tqdm(pd.read_csv('./chembl_33_molecules.tsv', sep='\t', chunksize=chunksize)):
        futures.append(executor.submit(process_chunk, chunk, pkidb_fingerprints))

    # Coletar os resultados de cada futuro
    for future in as_completed(futures):
        non_matching_results.append(future.result())

# Função para achatar uma lista de listas
def flatten_list_of_lists(lst):
    return [item for sublist in lst for item in sublist]

# Achatando os resultados em uma única lista
final_non_matching_smiles = flatten_list_of_lists(non_matching_results)

# Salvar os SMILES não correspondentes em um arquivo .tsv
pd.DataFrame(final_non_matching_smiles, columns=['canonical_smiles']).to_csv('./non_matching_smiles.tsv', sep='\t', index=False)
