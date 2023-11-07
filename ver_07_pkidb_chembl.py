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


    def find_similar_molecules(self, target_file_path, chembl_embeddings_path, threshold=0.8, batch_size=64, inner_batch_size=32):
        dtype = {'canonical_smiles': 'string', 'molregno': 'int32'}
        chembl_data = pd.read_csv(self.chembl_file_path, sep='\t', dtype=dtype)
        target_data = pd.read_csv(target_file_path, header=None, names=['target_smile'], dtype={'target_smile': 'string'})

        chembl_smiles = chembl_data['canonical_smiles'].tolist()
        target_smiles = target_data['target_smile'].tolist()

        similar_molecules_info = []
        similarity_scores = []

        # Tente carregar os embeddings do ChEMBL salvos previamente
        chembl_embeddings = self.load_embeddings(chembl_embeddings_path)
        if chembl_embeddings is None:
            # Se não existirem embeddings salvos, calcule-os e salve
            chembl_embeddings = self.get_molecule_embedding(chembl_smiles, inner_batch_size=inner_batch_size).to('cpu').numpy()
            self.save_embeddings(chembl_embeddings, chembl_embeddings_path)
        faiss.normalize_L2(chembl_embeddings)

        for i in tqdm(range(0, len(target_smiles), batch_size), desc="Processing target molecules"):
            target_batch = target_smiles[i:i + batch_size]
            target_embeddings = self.get_molecule_embedding(target_batch, inner_batch_size=inner_batch_size).to('cpu').numpy()  # Mova para a CPU antes de converter para numpy
            faiss.normalize_L2(target_embeddings)

            index = faiss.IndexFlatIP(chembl_embeddings.shape[1])
            index.add(chembl_embeddings)

            D, I = index.search(target_embeddings, k=chembl_embeddings.shape[0])

            for k, (indices, distances) in enumerate(zip(I, D)):
                for idx, score in zip(indices, distances):
                    if score > threshold:
                        similar_molecules_info.append({
                            'molregno': chembl_data.iloc[idx]['molregno'],
                            'chembl_smile': chembl_smiles[idx],
                            'target_smile': target_batch[k],
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


    @staticmethod
    def visualize_with_tsne(embeddings, labels):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis", s=100, alpha=0.7)
        plt.title('t-SNE Visualization of Molecule Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig('tsne_visualization_of_molecule_embeddings.png')
        plt.show()
        plt.close()


    @staticmethod
    def cluster_and_visualize(embeddings, num_clusters=5):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
        labels = kmeans.labels_

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis", s=100, alpha=0.7)
        plt.title('Clustered Visualization of Molecule Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig('clustered_visualization_of_molecule_embeddings.png')
        plt.show()
        plt.close()


    @staticmethod
    def save_similar_molecules_to_tsv(similar_molecules_info, file_path):
        df = pd.DataFrame(similar_molecules_info)
        df.to_csv(file_path, sep='\t', index=False)


def main():
    # Iniciar o cronômetro
    start_time = time.time()


    chembl_file_path = '/content/drive/MyDrive/chemBERTA/mol_10000.tsv'
    pkidb_file_path = '/content/drive/MyDrive/chemBERTA/smiles_from_pkidb_to_rdkit.tsv'
    chembl_embeddings_path = '/content/drive/MyDrive/chemBERTA/XX_TIMER_ver7_chembl_embeddings.npy'
    output_file_path = '/content/drive/MyDrive/chemBERTA/similar_molecules.tsv'
    threshold = 0.8

