import torch
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm

# Escolha o dispositivo disponível (CUDA ou CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path, pkidb_file_path, device="cpu"):
        self.chembl_file_path = chembl_file_path
        self.pkidb_file_path = pkidb_file_path
        self.device = device
        
        # Carregar o modelo ChemBERTa e o tokenizador
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        
        # Mova o modelo para o dispositivo correto
        self.model.to(device)
        
    def load_data(self, file_path):
        # Carregar as representações SMILES das moléculas
        data = pd.read_csv(file_path, sep='\t', header=None)
        smiles = data[0].tolist()
        return smiles
    
    def get_molecule_embedding(self, smiles):
        # Tokenizar a sequência SMILES
        tokens = self.tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Mova os tokens para o dispositivo correto
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        
        # Obter a saída do modelo
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        # Obter os embeddings das moléculas (última camada oculta)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    
    def find_similar_molecules(self):
        # Carregar os dados
        chembl_smiles = self.load_data(self.chembl_file_path)
        pkidb_smiles = self.load_data(self.pkidb_file_path)
        
        # Calcular os embeddings para as moléculas ChEMBL
        chembl_embeddings = []
        for smiles in tqdm(chembl_smiles, desc='Calculating ChEMBL embeddings', unit='mol'):
            embedding = self.get_molecule_embedding(smiles).cpu().numpy().flatten()
            chembl_embeddings.append(embedding)

        # Calcular os embeddings para as moléculas PKIDB
        pkidb_embeddings = []
        for smiles in tqdm(pkidb_smiles, desc='Calculating PKIDB embeddings', unit='mol'):
            embedding = self.get_molecule_embedding(smiles).cpu().numpy().flatten()
            pkidb_embeddings.append(embedding)

        # Calcular a similaridade cosseno entre as moléculas
        similarity_matrix = []
        for chembl_emb in tqdm(chembl_embeddings, desc='Calculating similarities', unit='pair'):
            similarities = [1 - cosine(chembl_emb, pkidb_emb) for pkidb_emb in pkidb_embeddings]
            similarity_matrix.append(similarities)
        
        return similarity_matrix
