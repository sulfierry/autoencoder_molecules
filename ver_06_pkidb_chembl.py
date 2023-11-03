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

# Definindo o dispositivo como GPU (CUDA) se disponível, senão será CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path, pkidb_file_path, device=device):
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

    def find_similar_molecules(self, threshold=0.8, batch_size=64, inner_batch_size=32):
        dtype = {'canonical_smiles': 'string', 'molregno': 'int32'}
        chembl_data = pd.read_csv(self.chembl_file_path, sep='\t', dtype=dtype)
        pkidb_data = pd.read_csv(self.pkidb_file_path, header=None, names=['pkidb_smile'], dtype={'pkidb_smile': 'string'})

        chembl_smiles = chembl_data['canonical_smiles'].tolist()
        pkidb_smiles = pkidb_data['pkidb_smile'].tolist()

        similar_molecules_info = []
        similarity_scores = []

        for i in tqdm(range(0, len(pkidb_smiles), batch_size), desc="Processing PKIDB molecules"):
            pkidb_batch = pkidb_smiles[i:i + batch_size]
            pkidb_embeddings = self.get_molecule_embedding(pkidb_batch, inner_batch_size=inner_batch_size).to('cpu').numpy()  # Mova para a CPU antes de converter para numpy
            faiss.normalize_L2(pkidb_embeddings)

            for j in tqdm(range(0, len(chembl_smiles), batch_size), desc="Processing ChEMBL molecules", leave=False):
                chembl_batch = chembl_smiles[j:j + batch_size]
                chembl_embeddings = self.get_molecule_embedding(chembl_batch, inner_batch_size=inner_batch_size).to('cpu').numpy()  # Mova para a CPU antes de converter para numpy
                faiss.normalize_L2(chembl_embeddings)

                index = faiss.IndexFlatIP(chembl_embeddings.shape[1])
                index.add(chembl_embeddings)

                D, I = index.search(pkidb_embeddings, k=chembl_embeddings.shape[0])

                for k, (indices, distances) in enumerate(zip(I, D)):
                    for idx, score in zip(indices, distances):
                        if score > threshold:
                            similar_molecules_info.append({
                                'molregno': chembl_data.iloc[j + idx]['molregno'],
                                'chembl_smile': chembl_batch[idx],
                                'pkidb_smile': pkidb_batch[k],
                                'similarity': score
                            })
                            similarity_scores.append(score)

        return similarity_scores, similar_molecules_info


class MoleculeVisualization:
    @staticmethod
    def plot_histogram(similarity_scores):
        plt.figure(figsize=(10, 6))
        plt.hist(similarity_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("Distribuição dos Scores de Similaridade")
        plt.xlabel("Score de Similaridade")
        plt.ylabel("Número de Moléculas")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('distribution_of_similarity_scores.png')
        plt.show()
        plt.close()
