# Parte 1: Importar Bibliotecas e Carregar os Dados

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

# Carregar o conjunto de dados ChEMBL
file_path_chembl = './molecules_with_bio_activities.tsv'
df_chembl = pd.read_csv(file_path_chembl, sep='\t')

# Carregar o conjunto de dados PKIDB
file_path_pkidb = './PKIDB/pkidb_2023-06-30.tsv'
df_pkidb = pd.read_csv(file_path_pkidb, sep='\t')

# Parte 2: Pré-processar os Dados SMILES do ChEMBL
# Calcular o comprimento de cada string SMILES
smiles_lengths = df_chembl['canonical_smiles'].apply(len)

# Filtrar o conjunto de dados para manter apenas moléculas com comprimento de SMILES <= 121 (percentil 95)
threshold_95 = smiles_lengths.quantile(0.95)
filtered_df_chembl = df_chembl[smiles_lengths <= threshold_95]

# Obter todos os caracteres únicos nas strings SMILES
unique_chars = set(''.join(filtered_df_chembl['canonical_smiles']))
char_to_index = {char: i for i, char in enumerate(sorted(unique_chars))}

# Função para converter uma string SMILES em uma lista de índices inteiros
def smiles_to_indices(smiles, char_to_index):
    return [char_to_index[char] for char in smiles]

# Converter todas as strings SMILES filtradas em listas de índices inteiros
smiles_indices = [smiles_to_indices(smiles, char_to_index) for smiles in filtered_df_chembl['canonical_smiles']]

# Converter listas de índices em tensores PyTorch e preencher sequências para terem o mesmo comprimento
smiles_tensors = [torch.tensor(indices, dtype=torch.long) for indices in smiles_indices]
smiles_padded = rnn_utils.pad_sequence(smiles_tensors, batch_first=True)
