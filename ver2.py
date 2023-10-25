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


class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path, pkidb_file_path, embedding_dim=10, latent_dim=32, num_epochs=10, batch_size=32):
        self.chembl_file_path = chembl_file_path
        print("Dados ChEMBL carregados com sucesso.")

        self.pkidb_file_path = pkidb_file_path
        print("Dados PKIDB carregados com sucesso.")
        
        self.dense_layer = nn.Linear(latent_dim, embedding_dim)  # Camada densa adicionada
        self.layer_norm = nn.LayerNorm(latent_dim)  # Camada de normalização adicionada

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
        smiles_tensors = [torch.tensor(indices, dtype=torch.long) for indices in smiles_indices]  # Alterado para float
        smiles_padded = rnn_utils.pad_sequence(smiles_tensors, batch_first=True, padding_value=char_to_index["<pad>"])

        print("SMILES convertidos para tensor com sucesso.")

        return smiles_padded

    def convert_indices_to_embeddings(self, indices_tensor):
        with torch.no_grad():
            embeddings = self.model.embedding(indices_tensor.long())  # Adicionado .long() para converter para inteiros
  
        print("Índices convertidos para embeddings com sucesso.")

        return embeddings
        

    def define_and_train_autoencoder(self, smiles_padded, vocab_size):
        print("Definindo e treinando o autoencoder...")
        # Definir a arquitetura do autoencoder
        self.model = Autoencoder(vocab_size, self.embedding_dim, self.latent_dim)

        # Definir critério de perda e otimizador
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Agendador de taxa de aprendizado

        # Preparação para treinamento com mini-batches
        dataset = torch.utils.data.TensorDataset(smiles_padded)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Treinar o modelo
        latent_features = []
        for epoch in range(self.num_epochs):
            total_loss = 0
            for smiles_batch in tqdm(data_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                smiles_batch = smiles_batch[0]
                smiles_batch = self.convert_indices_to_embeddings(smiles_batch)  # Adicionado esta linha
                optimizer.zero_grad()
                reconstructed, z = self.model(smiles_batch)
                loss = criterion(reconstructed, smiles_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                latent_features.append(z.detach().numpy())
                scheduler.step()  # Ajustar taxa de aprendizado a cada época

    
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, total_loss / len(data_loader)))
    
        print("Treinamento concluído.")
        
        return np.vstack(latent_features)
        
    def find_similar_molecules(self):
        print("Iniciando busca por moléculas similares...")
 
        # Parte 2: Pré-processar os Dados SMILES do ChEMBL
        print("Preprocessando dados SMILES do ChEMBL...")

        self.filtered_df_chembl, self.char_to_index, threshold_95 = self.preprocess_smiles_data(self.df_chembl, 'canonical_smiles')
    
        vocab_size = len(self.char_to_index)
        print("Tamanho do vocabulário:", vocab_size)

        print("Convertendo SMILES para tensor...")
        try:
            smiles_padded = self.smiles_to_tensor(self.filtered_df_chembl, 'canonical_smiles', self.char_to_index, vocab_size)
            print("Conversão de SMILES para tensor concluída com sucesso.")
        except Exception as e:
            print("Ocorreu um erro durante a conversão de SMILES para tensor:", str(e))
            return
            
        # Verificar se os índices estão no intervalo válido
        max_index = smiles_padded.max().item()
        if max_index >= vocab_size:
            print("Erro: Índice máximo é maior ou igual ao tamanho do vocabulário.")
            return

        # Parte 3: Definir e Treinar o Autoencoder
        self.latent_features_chembl = self.define_and_train_autoencoder(smiles_padded, vocab_size)
        # Nota: Aqui, o treinamento é feito com o conjunto de dados ChEMBL, mas o mesmo modelo pode ser aplicado ao conjunto de dados PKIDB
    
        # Parte 4: Preprocessar e Converter os Dados SMILES do PKIDB
        print("Preprocessando e convertendo dados do PKIDB...")
        self.filtered_df_pkidb, _, _ = self.preprocess_smiles_data(self.df_pkidb, 'Canonical_Smiles')
        smiles_padded_pkidb = self.smiles_to_tensor(self.filtered_df_pkidb, 'Canonical_Smiles', self.char_to_index, vocab_size)
    
        # Parte 5: Obter Representações Latentes das Moléculas do PKIDB
        _, latent_features_pkidb = self.model(smiles_padded_pkidb)
    
        # Parte 6: Calcular Similaridade de Cosseno entre as Representações Latentes
        similarity_matrix = cosine_similarity(self.latent_features_chembl.detach().numpy(), latent_features_pkidb.detach().numpy())
        
        # Parte 7: Identificar Moléculas Similares
        similar_molecules = np.where(similarity_matrix > 0.95)  # Ajuste este valor conforme necessário
        for chembl_index, pkidb_index in zip(*similar_molecules):
            chembl_smiles = self.filtered_df_chembl.iloc[chembl_index]['canonical_smiles']
            pkidb_smiles = self.filtered_df_pkidb.iloc[pkidb_index]['Canonical_Smiles']
            print(f"Molécula similar encontrada: {chembl_smiles} (ChEMBL) e {pkidb_smiles} (PKIDB)")
            
        print("Busca por moléculas similares concluída.")


class Autoencoder(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_size-1)
        self.encoder = nn.LSTM(embedding_dim, latent_dim, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, embedding_dim, batch_first=True)
        self.linear = nn.Linear(embedding_dim, vocab_size)  # Adicionado
        
    def forward(self, x):
        if x.max().item() >= self.embedding.num_embeddings:
            raise IndexError("Índice fora do alcance. Certifique-se de que todos os índices são menores que o tamanho do vocabulário.")
        x = x.long()  # Garantindo que x é do tipo long
        x_embedded = self.embedding(x)
        _, (h_n, _) = self.encoder(x_embedded)
        z = h_n.squeeze(0)
        
        z = F.relu(self.dense_layer(z))  # Aplicando transformação linear e função de ativação
        z = self.layer_norm(z)  # Aplicando normalização da camada
        
        # Repita z para ter a mesma sequência de comprimento que x
        z_repeated = z.unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decodifique
        x_reconstructed, _ = self.decoder(z_repeated)
        
        # Transformar de volta para o espaço original
        x_reconstructed = self.linear(x_reconstructed)

        return x_reconstructed, z


if __name__ == "__main__":
    # Certifique-se de que o caminho até os arquivos está correto
    # chembl_file_path = "path/to/chembl_file.tsv"
    # pkidb_file_path = "path/to/pkidb_file.tsv"
    chembl_file_path = './molecules_with_bio_activities.tsv'
    pkidb_file_path = './PKIDB/pkidb_2023-06-30.tsv'

    molecule_similarity_finder = MoleculeSimilarityFinder(chembl_file_path, pkidb_file_path)
    molecule_similarity_finder.find_similar_molecules()

