import torch
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Escolha o dispositivo disponível (CUDA ou CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path, pkidb_file_path, device=device):
        self.chembl_file_path = chembl_file_path
        self.pkidb_file_path = pkidb_file_path
        self.device = device
        
        # Carregar o modelo ChemBERTa e o tokenizador
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1').to(device).half()
        
    def load_data(self, file_path):
        data = pd.read_csv(file_path, sep='\t', header=None, dtype={0: 'str'})
        smiles = data[0].tolist()
        return smiles
    
    def get_molecule_embedding(self, smiles_list):
        tokens = self.tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    
    def find_similar_molecules(self, threshold=0.8):
        chembl_smiles = self.load_data(self.chembl_file_path)
        pkidb_smiles = self.load_data(self.pkidb_file_path)
        
        batch_size = 256
        similarity_matrix = np.zeros((len(chembl_smiles), len(pkidb_smiles)))

        # Processa as moléculas ChEMBL em batches
        for i in tqdm(range(0, len(chembl_smiles), batch_size), desc='Processing ChEMBL molecules'):
            chembl_batch = chembl_smiles[i:i + batch_size]
            chembl_embeddings = self.get_molecule_embedding(chembl_batch).cpu().numpy()
            
            # Processa as moléculas PKIDB em batches
            for j in range(0, len(pkidb_smiles), batch_size):
                pkidb_batch = pkidb_smiles[j:j + batch_size]
                pkidb_embeddings = self.get_molecule_embedding(pkidb_batch).cpu().numpy()

                # Calcula a similaridade cosseno entre os dois batches
                similarities = 1 - cdist(chembl_embeddings, pkidb_embeddings, metric='cosine')
                similarity_matrix[i:i+batch_size, j:j+batch_size] = similarities

        # Filtrar a matriz de similaridade para manter apenas os pares com uma pontuação de similaridade acima do limiar
        similarity_matrix[similarity_matrix < threshold] = 0
        return similarity_matrix

def main():
    print("Using device:", device)
    # Caminhos para os arquivos de entrada
    chembl_file_path = '/content/molecules_with_bio_activities.tsv'
    pkidb_file_path = '/content/pkidb_2023-06-30.tsv'
    
    # Limiar de similaridade
    similarity_threshold = 0.8
    
    # Instancie a classe e encontre moléculas similares
    similarity_finder = MoleculeSimilarityFinder(chembl_file_path, pkidb_file_path)
    similarity_matrix = similarity_finder.find_similar_molecules(similarity_threshold)
    
    print("Similarity Matrix:")
    print(similarity_matrix)

if __name__ == "__main__":
    main()




