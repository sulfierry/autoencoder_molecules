import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device =  "cuda" if torch.cuda.is_available() else "cpu" #"mps" if torch.backends.mps.is_available() else
torch.autograd.set_detect_anomaly(True)


class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path, pkidb_file_path, embedding_dim=128, 
                 latent_dim=64, num_epochs=10, batch_size=32, device=device,learning_rate=0.001):
        self.device = device
        self.chembl_file_path = chembl_file_path
        print("Dados ChEMBL carregados com sucesso.")
        self.pkidb_file_path = pkidb_file_path
        print("Dados PKIDB carregados com sucesso.")
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None
        self.char_to_index = None
        self.latent_features_chembl = None
        self.filtered_df_chembl = None
        self.filtered_df_pkidb = None

        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.df_chembl = self.load_data(self.chembl_file_path, 'canonical_smiles')
        self.df_pkidb = self.load_data(self.pkidb_file_path, 'Canonical_Smiles')

    def load_data(self, file_path, smiles_column):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")
        print(f"Carregando dados de {file_path}...")
        df = pd.read_csv(file_path, sep='\t')
        if smiles_column not in df.columns:
            raise ValueError(f"A coluna {smiles_column} não está presente no arquivo {file_path}.")
        print(f"Dados carregados com sucesso. {df.shape[0]} linhas e {df.shape[1]} colunas.")
        return df

    @staticmethod
    def preprocess_smiles_data(df, smiles_column):
        print("Preprocessando dados SMILES...")

        # Tratamento de valores nulos
        original_size = df.shape[0]
        df = df[df[smiles_column].notna()]
        print(f"Removidos {original_size - df.shape[0]} registros com valores nulos na coluna {smiles_column}.")

        # Calcular o comprimento de cada string SMILES
        smiles_lengths = df[smiles_column].apply(len)

        # Filtrar o conjunto de dados para manter apenas moléculas com comprimento de SMILES <= 121 (percentil 95)
        threshold_95 = smiles_lengths.quantile(0.95)
        filtered_df = df[smiles_lengths <= threshold_95]

        # Obter todos os caracteres únicos nas strings SMILES e adicionar um caractere de preenchimento
        unique_chars = set(''.join(filtered_df[smiles_column]))
        unique_chars.add("<pad>")  # Adicionando caractere de preenchimento
        char_to_index = {char: i for i, char in enumerate(sorted(unique_chars))}

        print(f"Conjunto de dados filtrado para manter moléculas com comprimento de SMILES <= {threshold_95}.")
        print("Tamanho do vocabulário:", len(char_to_index))
        print("Dados SMILES preprocessados com sucesso.")

        return filtered_df, char_to_index, threshold_95

    def smiles_to_tensor(self, df, smiles_column, char_to_index, vocab_size):
        print("Convertendo SMILES para tensor...")
        smiles_indices = []
        for smiles in df[smiles_column]:
            indices = [char_to_index.get(char, char_to_index["<pad>"]) for char in smiles]
            smiles_indices.append(indices)

        smiles_tensors = [torch.tensor(indices, dtype=torch.long) for indices in smiles_indices]
        smiles_padded = rnn_utils.pad_sequence(smiles_tensors, batch_first=True, padding_value=char_to_index["<pad>"])

        print("SMILES convertidos para tensor com sucesso.")
        return smiles_padded

    def convert_indices_to_embeddings(self, indices_tensor):
        with torch.no_grad():
            embeddings = self.model.embedding(indices_tensor.long())
        print("Índices convertidos para embeddings com sucesso.")
        return embeddings

    def define_and_train_autoencoder(self, smiles_padded, vocab_size):
        print("Definindo e treinando o autoencoder...")
        dataset = TensorDataset(smiles_padded, smiles_padded)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = Autoencoder(vocab_size, self.embedding_dim, self.latent_dim).to(self.device)
        self.model.apply(self.init_weights)  # Inicializando os pesos

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)  # Adicionando regularização L2
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Ajustando o agendador de taxa de aprendizado
        
        latent_features = []
        for epoch in range(self.num_epochs):
            total_loss = 0
            for smiles_batch, _ in tqdm(data_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}', file=sys.stdout):
                smiles_batch = smiles_batch.to(self.device)
                self.model.train()
                
                reconstructed, z = self.model(smiles_batch)
                reconstructed = reconstructed.view(-1, reconstructed.size(-1))
                smiles_batch = smiles_batch.view(-1)
                
                optimizer.zero_grad()
                loss = criterion(reconstructed, smiles_batch)
                loss.backward()
                
                optimizer.step()
                total_loss += loss.item()
                latent_features.append(z.detach().cpu().numpy())
                
            print(f"\nLoss no Epoch {epoch+1}: {total_loss / len(data_loader)}")
            scheduler.step()
            
        print("Autoencoder treinado com sucesso.")
        return np.concatenate(latent_features)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


    def find_similar_molecules(self):
        self.filtered_df_chembl, self.char_to_index, threshold_95 = self.preprocess_smiles_data(self.df_chembl, 'canonical_smiles')
        vocab_size = len(self.char_to_index)
        smiles_padded_chembl = self.smiles_to_tensor(self.filtered_df_chembl, 'canonical_smiles', self.char_to_index, vocab_size)
        self.latent_features_chembl = self.define_and_train_autoencoder(smiles_padded_chembl, vocab_size)
        filtered_df_pkidb, _, _ = self.preprocess_smiles_data(self.df_pkidb, 'Canonical_Smiles')
        smiles_padded_pkidb = self.smiles_to_tensor(filtered_df_pkidb, 'Canonical_Smiles', self.char_to_index, vocab_size)
        latent_features_pkidb = self.convert_indices_to_embeddings(smiles_padded_pkidb.to(self.device)).cpu().numpy()

        # Ajustando a dimensão do tensor antes de calcular a similaridade de cosseno
        latent_features_chembl_reshaped = self.latent_features_chembl.reshape(self.latent_features_chembl.shape[0], -1)
        latent_features_pkidb_reshaped = latent_features_pkidb.reshape(latent_features_pkidb.shape[0], -1)

        similarity_matrix = cosine_similarity(latent_features_pkidb_reshaped, latent_features_chembl_reshaped)
        return similarity_matrix


class Autoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, latent_dim, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, embedding_dim, batch_first=True)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

    def forward(self, x):
        embeddings = self.embedding(x)
        _, (hidden, _) = self.encoder(embeddings)
        z = hidden.squeeze(0)
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        decoder_output, _ = self.decoder(z)
        logits = self.fc(decoder_output)
        return F.log_softmax(logits, dim=-1), z




chembl_file_path = '/content/molecules_with_bio_activities.tsv'
pkidb_file_path = '/content/pkidb_2023-06-30.tsv'
similarity_finder = MoleculeSimilarityFinder(chembl_file_path, pkidb_file_path, device=device)
similarity_matrix = similarity_finder.find_similar_molecules()
