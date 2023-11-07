import os
import time
import torch
import faiss
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from transformers import RobertaModel, RobertaTokenizer
import os

# Definindo o dispositivo como GPU (CUDA) se disponível, senão será CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path=None, pkidb_file_path=None, device='cpu'):
        self.chembl_file_path = chembl_file_path
        self.pkidb_file_path = pkidb_file_path
        self.device = device

        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = torch.nn.DataParallel(self.model).to(device)

    def get_molecule_embedding(self, smiles_list, inner_batch_size=32):
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(smiles_list), inner_batch_size):
                smiles_batch = smiles_list[i:i + inner_batch_size]
                tokens = self.tokenizer(smiles_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
                tokens = {key: val.to(self.device) for key, val in tokens.items()}
                outputs = self.model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings)  # Mantenha os embeddings na GPU
        return torch.cat(all_embeddings, dim=0)

    def save_embeddings(self, embeddings, file_path):
        np.save(file_path, embeddings)

    def load_embeddings(self, file_path):
        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            return None
