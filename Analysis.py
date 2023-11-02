import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFMCS, AllChem
from scipy.spatial.distance import pdist, squareform
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


class SmilesAnalysis:
    def __init__(self, data_iterator, smiles_col):
        self.data_iterator = data_iterator
        self.smiles_col = smiles_col
        self.mols = []
        self.fps = []
        self.process_chunks()

    def process_chunks(self):
        total_chunks = sum(1 for _ in self.data_iterator)  # Contar o número total de chunks
        self.data_iterator = pd.read_csv(data_path, sep='\t', chunksize=chunk_size)  # Reinicializar o iterador
        pbar = tqdm(total=total_chunks)
        for chunk in self.data_iterator:
            mols_chunk = chunk[self.smiles_col].dropna().apply(Chem.MolFromSmiles)
            fps_chunk = self.compute_fingerprints_parallel(mols_chunk, pbar)
            self.mols.extend(mols_chunk)
            self.fps.extend(fps_chunk)
            pbar.update(1)
        pbar.close()

    def compute_fingerprints_parallel(self, mols_chunk, pbar):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fps_chunk = list(tqdm(executor.map(self.compute_fingerprint, mols_chunk), total=len(mols_chunk), desc="Computing fingerprints", leave=False))
        pbar.set_postfix(refresh=True)  # Atualizar a barra de progresso dos chunks
        return [fp for fp in fps_chunk if fp is not None]

    @staticmethod
    def compute_fingerprint(mol):
        if mol is not None:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        return None

    def calculate_statistics(self):
        lengths = self.data[self.smiles_col].apply(len)
        return lengths.describe()

    def plot_histogram(self):
        lengths = self.data[self.smiles_col].apply(len)
        plt.figure(figsize=(10, 6))
        sns.histplot(lengths, bins=30, kde=True)
        plt.title('Distribuição do Tamanho das Estruturas Moleculares')
        plt.xlabel('Comprimento da Estrutura Molecular')
        plt.ylabel('Frequência')
        plt.show()

    def find_extremes(self):
        lengths = self.data[self.smiles_col].apply(len)
        max_idx = lengths.idxmax()
        min_idx = lengths.idxmin()
        return self.data.iloc[max_idx], self.data.iloc[min_idx]

    @staticmethod
    def visualize_with_tsne(embeddings, labels):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis", s=100, alpha=0.7)
        plt.title('t-SNE Visualization of Molecule Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

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
        plt.show()

    def cluster_analysis(self):
        embeddings = np.array(self.fps)
        labels = np.zeros(len(embeddings))  # Dummy labels for visualization
        self.visualize_with_tsne(embeddings, labels)
        self.cluster_and_visualize(embeddings)
