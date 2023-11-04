import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy.spatial.distance import pdist, squareform
import concurrent.futures
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans

# Configuração básica de logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG)

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
          fps_chunk = self.compute_fingerprints_parallel(mols_chunk)
          self.mols.extend(mols_chunk)
          self.fps.extend(fps_chunk)
          pbar.update(1)
        pbar.close()


    def compute_fingerprints_parallel(self, mols_chunk):
        # Limitar o número de workers para evitar sobrecarregar o sistema
        num_workers = min(4, os.cpu_count())
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            fps_chunk = list(executor.map(self.compute_fingerprint, mols_chunk))
        return [fp for fp in fps_chunk if fp is not None]

    @staticmethod
    def compute_fingerprint(mol):
        try:
            if mol is not None:
                return AllChem.GetMorganFingerprintAsBitVect(mol, 2)
            return None
        except Exception as e:
            # Logar a exceção
            logging.exception(f"Erro ao calcular a impressão digital: {e}")
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

class MolecularComparison:
    def __init__(self, data, smiles_col1, smiles_col2):
        self.data = data
        self.smiles_col1 = smiles_col1
        self.smiles_col2 = smiles_col2
        self.mols1 = data[smiles_col1].apply(Chem.MolFromSmiles)
        self.mols2 = data[smiles_col2].apply(Chem.MolFromSmiles)

    def tanimoto_similarity(self):
        fps1 = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in self.mols1 if m is not None]
        fps2 = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in self.mols2 if m is not None]
        similarities = [DataStructs.FingerprintSimilarity(fp1, fp2) for fp1, fp2 in zip(fps1, fps2)]
        self.data['tanimoto_similarity'] = similarities
        return similarities

    def chemical_space_visualization(self):
        fps1 = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in self.mols1 if m is not None]
        fps2 = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in self.mols2 if m is not None]
        embeddings = np.array(fps1 + fps2)
        labels = ['chembl'] * len(fps1) + ['pkidb'] * len(fps2)

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis", s=100, alpha=0.7)
        plt.title('Chemical Space Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(title='Source')
        plt.show()

    def chemical_diversity_analysis(self):
        fps1 = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in self.mols1 if m is not None]
        fps2 = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in self.mols2 if m is not None]

        diversity1 = -np.mean(squareform(pdist(np.array(fps1), metric='jaccard')))
        diversity2 = -np.mean(squareform(pdist(np.array(fps2), metric='jaccard')))
        inter_diversity = -np.mean(pdist(np.vstack([fps1, fps2]), metric='jaccard'))

        return diversity1, diversity2, inter_diversity

    @staticmethod
    def cluster_and_visualize(embeddings, num_clusters=5):
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
        labels = kmeans.labels_

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis", s=100, alpha=0.7)
        plt.title('Clustered Visualization of Molecule Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

if __name__ == "__main__":
    chunk_size = 500  # Defina o tamanho do lote com base na sua memória disponível
    data_path = '/content/similar_molecules_3.tsv'

    # Análise para 'chembl_smile' e 'pkidb_smile'
    data_iterator = pd.read_csv(data_path, sep='\t', chunksize=chunk_size)
    chembl_analysis = SmilesAnalysis(data_iterator, 'chembl_smile')
    print("Estatísticas do Tamanho das Estruturas (chembl_smile):")
    print(chembl_analysis.calculate_statistics())
    chembl_analysis.plot_histogram()

    data_iterator = pd.read_csv(data_path, sep='\t', chunksize=chunk_size)
    pkidb_analysis = SmilesAnalysis(data_iterator, 'pkidb_smile')
    print("Estatísticas do Tamanho das Estruturas (pkidb_smile):")
    print(pkidb_analysis.calculate_statistics())
    pkidb_analysis.plot_histogram()

    # Comparação Molecular
    data = pd.read_csv(data_path, sep='\t')
    molecular_comparison = MolecularComparison(data, 'chembl_smile', 'pkidb_smile')
    print("Similaridades de Tanimoto:")
    similarities = molecular_comparison.tanimoto_similarity()
    print(similarities)

    # Visualização do Espaço Químico
    molecular_comparison.chemical_space_visualization()

    # Análise de Diversidade Química
    diversity1, diversity2, inter_diversity = molecular_comparison.chemical_diversity_analysis()
    print("Diversidade Química (chembl_smile):", diversity1)
    print("Diversidade Química (pkidb_smile):", diversity2)
    print("Diversidade Química Inter-conjuntos:", inter_diversity)
