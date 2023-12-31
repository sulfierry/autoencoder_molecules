import pandas as pd
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

def process_chunk(chunk, pkidb_fingerprints, pkidb_smiles):
    chunk_fingerprints = chunk['canonical_smiles'].apply(smiles_to_fingerprint)
    non_matching_smiles_pkidb = []
    for pkidb_fp, pkidb_smile in zip(pkidb_fingerprints, pkidb_smiles):
        if pkidb_fp is not None and not any(DataStructs.FingerprintSimilarity(pkidb_fp, chembl_fp) == 1.0 for chembl_fp in chunk_fingerprints if chembl_fp is not None):
            non_matching_smiles_pkidb.append(pkidb_smile)
    return non_matching_smiles_pkidb

# Carregar os SMILES do pkidb
pkidb_smiles = pd.read_csv('./pkidb_2023-06-30.tsv', sep='\t')
pkidb_smiles = pkidb_smiles.dropna(subset=['Canonical_Smiles'])
pkidb_smiles['fingerprint'] = pkidb_smiles['Canonical_Smiles'].apply(smiles_to_fingerprint).dropna()
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
