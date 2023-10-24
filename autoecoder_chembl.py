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

# Parte 3: Definir e Treinar o Autoencoder

# Definir a arquitetura do autoencoder
class Autoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # Aggregating embeddings
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

# Instanciar o modelo
embedding_dim = 10
model = Autoencoder(len(char_to_index), embedding_dim, latent_dim=32)

# Definir critério de perda e otimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinar o modelo
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    reconstructed, z = model(smiles_padded)
    loss = criterion(reconstructed, model.embedding(smiles_padded).mean(dim=1))
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Extrair características latentes
latent_features_chembl = z.detach().numpy()



# Parte 4: Pré-processar os Dados SMILES do PKIDB e Calcular as Características Latentes

# Calcular o comprimento de cada string SMILES no PKIDB
smiles_lengths_pkidb = df_pkidb['Canonical_Smiles'].apply(len)

# Filtrar o conjunto de dados para manter apenas moléculas com um comprimento de SMILES adequado
filtered_df_pkidb = df_pkidb[smiles_lengths_pkidb <= threshold_95]  # Usando o mesmo threshold do ChEMBL

# Converter todas as strings SMILES filtradas do PKIDB em listas de índices inteiros
# (Se algum caractere não estiver presente em char_to_index, ele será ignorado)
pkidb_smiles_indices = [[char_to_index[char] for char in smiles if char in char_to_index] for smiles in filtered_df_pkidb['Canonical_Smiles']]

# Converter listas de índices em tensores PyTorch e preencher sequências para terem o mesmo comprimento
pkidb_smiles_tensors = [torch.tensor(indices, dtype=torch.long) for indices in pkidb_smiles_indices]
pkidb_smiles_padded = rnn_utils.pad_sequence(pkidb_smiles_tensors, batch_first=True)

# Certifique-se de que todos os índices em pkidb_smiles_padded estão dentro do alcance do vocabulário do modelo
vocab_size = model.embedding.num_embeddings  # Obtenha o tamanho do vocabulário do modelo
pkidb_smiles_padded = torch.clamp(pkidb_smiles_padded, 0, vocab_size - 1)

# Calcular as características latentes para o conjunto de dados PKIDB
with torch.no_grad():
    model.eval()  # Colocar o modelo em modo de avaliação
    _, z_pkidb = model(pkidb_smiles_padded)
    latent_features_pkidb = z_pkidb.numpy()

