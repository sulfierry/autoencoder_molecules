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

class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path, pkidb_file_path, embedding_dim=10, latent_dim=32, num_epochs=10, batch_size=32):
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
        
        # Carregar dados
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
        # Verifique se todos os caracteres estão no vocabulário
        print("Verificando se todos os caracteres estão no vocabulário...")
        all_chars = set(''.join(df[smiles_column]))
        missing_chars = all_chars - set(char_to_index.keys())
        if missing_chars:
            print("Atenção: Os seguintes caracteres não estão no vocabulário e serão ignorados:", missing_chars)

        # Converter todas as strings SMILES filtradas em listas de índices inteiros
        smiles_indices = []
        for smiles in df[smiles_column]:
            indices = [char_to_index[char] for char in smiles if char in char_to_index]
            smiles_indices.append(indices)
    
        # Converter listas de índices em tensores PyTorch e preencher sequências para terem o mesmo comprimento
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
    
        # Definir o dataset e o DataLoader
        dataset = TensorDataset(smiles_padded)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Definir o modelo
        self.model = Autoencoder(vocab_size, self.embedding_dim, self.latent_dim)
        
        # Definir a função de perda e o otimizador
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorar o índice de preenchimento na perda
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Opcional: Definir um scheduler para ajustar a taxa de aprendizado
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        # Treinar o modelo
        latent_features = []
        for epoch in range(self.num_epochs):
            total_loss = 0
            for smiles_batch in tqdm(data_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                smiles_batch = smiles_batch[0]
                
                # Garanta que as operações estejam no modo de treinamento
                self.model.train()
                reconstructed, z = self.model(smiles_batch)
                
                # Ajustar tamanho das dimensões, se necessário
                if reconstructed.shape[1] != smiles_batch.shape[1]:
                    reconstructed = reconstructed[:, :smiles_batch.shape[1], :]
    
                # Aplique view para corresponder às dimensões esperadas pela CrossEntropyLoss
                smiles_batch = smiles_batch.view(-1)
                reconstructed = reconstructed.view(-1, reconstructed.size(-1))
    
                # Agora, você pode calcular a perda já que os tensores têm a mesma forma
                loss = criterion(reconstructed, smiles_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                latent_features.append(z.cpu().detach().numpy())
    
            scheduler.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, total_loss / len(data_loader)))
        
        # Salvar os recursos latentes
        latent_features = np.concatenate(latent_features, axis=0)
        print("Autoencoder treinado com sucesso.")
        return latent_features
    
    
    def find_similar_molecules(self, latent_features_chembl, latent_features_pkidb, n_top_molecules=5):
        print("Encontrando moléculas similares...")
        
        # Calcular as semelhanças de cosseno
        similarities = cosine_similarity(latent_features_chembl, latent_features_pkidb)
        
        # Encontrar as moléculas mais similares
        top_molecule_indices = np.argsort(similarities, axis=0)[-n_top_molecules:]
        
        print("Moléculas similares encontradas com sucesso.")
       
        return top_molecule_indices    
       
    def run(self):
        print("Iniciando o processo de busca por similaridade de moléculas...")
        
        # Preprocessar os dados SMILES
        self.filtered_df_chembl, self.char_to_index, max_len = self.preprocess_smiles_data(self.df_chembl, 'canonical_smiles')
        self.filtered_df_pkidb, _, _ = self.preprocess_smiles_data(self.df_pkidb, 'Canonical_Smiles')

        # Converter os dados SMILES em tensores
        vocab_size = len(self.char_to_index)
        smiles_padded_chembl = self.smiles_to_tensor(self.filtered_df_chembl, 'canonical_smiles', self.char_to_index, vocab_size)
        smiles_padded_pkidb = self.smiles_to_tensor(self.filtered_df_pkidb, 'Canonical_Smiles', self.char_to_index, vocab_size)

        # Definir e treinar o autoencoder
        latent_features_chembl = self.define_and_train_autoencoder(smiles_padded_chembl, vocab_size)
        latent_features_pkidb = self.define_and_train_autoencoder(smiles_padded_pkidb, vocab_size)

        # Encontrar moléculas similares
        top_molecule_indices = self.find_similar_molecules(latent_features_chembl, latent_features_pkidb)
        
        print("Processo concluído com sucesso.")
        return top_molecule_indices

class Autoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, latent_dim, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, embedding_dim, batch_first=True)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        embeddings = self.embedding(x)
        _, (hn, _) = self.encoder(embeddings)
        reconstructions, _ = self.decoder(hn.repeat(1, x.size(1), 1))
        reconstructions = reconstructions.contiguous().view(-1, reconstructions.size(-1))
        reconstructions = self.linear(reconstructions)
        reconstructions = reconstructions.view(x.size(0), -1, reconstructions.size(-1))
        return reconstructions, hn.squeeze(0)



if __name__ == "__main__":
    chembl_file_path = './molecules_with_bio_activities.tsv'
    pkidb_file_path = './PKIDB/pkidb_2023-06-30.tsv'
    finder = MoleculeSimilarityFinder(chembl_file_path, pkidb_file_path)
    top_molecule_indices = finder.run()
    print("Índices das moléculas mais similares encontradas:", top_molecule_indices)


