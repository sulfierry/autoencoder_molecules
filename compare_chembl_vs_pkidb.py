# compara os fingerprints do pkidb com cada chunk do ChEMBL, acumulando os SMILES não correspondentes do pkidb.
    
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

def process_chunk(chunk, pkidb_fingerprints, pkidb_smiles):
    non_matching_smiles_pkidb = []
    for pkidb_fp, pkidb_smile in zip(pkidb_fingerprints, pkidb_smiles):
        if not any(DataStructs.FingerprintSimilarity(pkidb_fp, smiles_to_fingerprint(row['canonical_smiles'])) == 1.0 for _, row in chunk.iterrows() if pkidb_fp is not None):
            non_matching_smiles_pkidb.append(pkidb_smile)
    return non_matching_smiles_pkidb

# Carregar os SMILES do pkidb
pkidb_smiles = pd.read_csv('./pkidb_2023-06-30.tsv', sep='\t')
pkidb_smiles['fingerprint'] = pkidb_smiles['Canonical_Smiles'].apply(smiles_to_fingerprint)
pkidb_fingerprints = pkidb_smiles['fingerprint'].tolist()
pkidb_smiles_list = pkidb_smiles['Canonical_Smiles'].tolist()

# Utilizar todas as CPUs disponíveis
cpu_count = os.cpu_count()

# Processar o arquivo ChEMBL em partes
chunksize = 10240
non_matching_results_pkidb = []

with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    futures = []
    for chunk in tqdm(pd.read_csv('./chembl_33_molecules.tsv', sep='\t', chunksize=chunksize)):
        futures.append(executor.submit(process_chunk, chunk, pkidb_fingerprints, pkidb_smiles_list))

    for future in as_completed(futures):
        non_matching_results_pkidb.extend(future.result())

# Salvar os SMILES não correspondentes do pkidb em um arquivo .tsv
pd.DataFrame(non_matching_results_pkidb, columns=['Canonical_Smiles']).to_csv('./non_matching_smiles_pkidb.tsv', sep='\t', index=False)
