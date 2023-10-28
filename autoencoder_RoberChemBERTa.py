import torch
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Escolha o dispositivo disponível (CUDA ou CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path, pkidb_file_path, device=device):
        self.chembl_file_path = chembl_file_path
        self.pkidb_file_path = pkidb_file_path
        self.device = device

        # Carregar o modelo ChemBERTa e o tokenizador
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')

        # Mova o modelo para o dispositivo correto
        self.model.to(device)

    def load_data(self, file_path):
        data = pd.read_csv(file_path, sep='\t', header=None)
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
        """Encontra moléculas similares entre ChemBL e PKIDB."""
        chembl_smiles = self.load_data(self.chembl_file_path)
        pkidb_smiles = self.load_data(self.pkidb_file_path)

        similar_molecules_info = []  # Lista para armazenar informações sobre moléculas similares
        similarity_scores = []  # lista para armazenar as pontuações de similaridade

        batch_size = 1024

        # Processa as moléculas ChEMBL em batches
        for i in tqdm(range(0, len(chembl_smiles), batch_size), desc='Processing ChEMBL molecules'):
            chembl_batch = chembl_smiles[i:i + batch_size]
            chembl_embeddings = self.get_molecule_embedding(chembl_batch).cpu().numpy()

            # Processa as moléculas PKIDB em batches
            for j in range(0, len(pkidb_smiles), batch_size):
                pkidb_batch = pkidb_smiles[j:j + batch_size]
                pkidb_embeddings = self.get_molecule_embedding(pkidb_batch).cpu().numpy()

                # Calcula a similaridade cosseno entre os dois batches
                for chembl_idx, chembl_emb in enumerate(chembl_embeddings):
                    for pkidb_idx, pkidb_emb in enumerate(pkidb_embeddings):
                        similarity_score = 1 - cosine(chembl_emb, pkidb_emb)
                        if similarity_score > threshold:
                            similar_molecules_info.append({
                                'chembl_smile': chembl_batch[chembl_idx],
                                'pkidb_smile': pkidb_batch[pkidb_idx],
                                'similarity': similarity_score
                            })
                            similarity_scores.append(similarity_score)

        return similarity_scores, similar_molecules_info

    def visualize_with_tsne(self, similarity_matrix, similarity_scores):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(similarity_matrix)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=similarity_scores, cmap="YlGnBu")
        plt.colorbar()
        plt.show()

def plot_histogram(similarity_scores):
    plt.hist(similarity_scores, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Number of Molecules")
    plt.show()

# Restante do código (funções save_similar_molecules_to_tsv e plot_similarity_matrix) permanecem iguais

def save_similar_molecules_to_tsv(similar_molecules_info, file_path):
    df = pd.DataFrame(similar_molecules_info)
    df.to_csv(file_path, sep='\t', index=False)

def plot_similarity_matrix(similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix.todense(), cmap="YlGnBu", vmin=0, vmax=1)
    plt.show()

