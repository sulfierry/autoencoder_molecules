import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from concurrent.futures import ProcessPoolExecutor
import os

def process_chunk(chunk, pkidb_fingerprints):
    non_matching_smiles = []
    for _, row in chunk.iterrows():
        chembl_fp = smiles_to_fingerprint(row['canonical_smiles'])
        if not any(DataStructs.FingerprintSimilarity(chembl_fp, fp) == 1.0 for fp in pkidb_fingerprints):
            non_matching_smiles.append(row['canonical_smiles'])
    return non_matching_smiles


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)



# Carregar os SMILES do pkidb
pkidb_smiles = pd.read_csv('/mnt/data/pkidb_2023-06-30.tsv', sep='\t')
pkidb_smiles['fingerprint'] = pkidb_smiles['Canonical_Smiles'].apply(smiles_to_fingerprint)

# Preparar a lista de fingerprints do pkidb para comparação
pkidb_fingerprints = pkidb_smiles['fingerprint'].tolist()

# Processar o arquivo ChEMBL em partes
chunksize = 10000  # Ajuste conforme necessário
non_matching_results = []

# Utilizar todas as CPUs disponíveis
cpu_count = os.cpu_count()



with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    futures = []
    for chunk in pd.read_csv('/mnt/data/chembl_33_molecules.tsv', sep='\t', chunksize=chunksize):
        futures.append(executor.submit(process_chunk, chunk, pkidb_fingerprints))
    
    for future in futures:
        non_matching_results.extend(future.result())


# Concatenar todos os resultados não correspondentes
final_non_matching_smiles = np.concatenate(non_matching_results)
