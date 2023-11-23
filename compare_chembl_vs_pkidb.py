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

