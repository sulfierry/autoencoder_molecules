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
    
