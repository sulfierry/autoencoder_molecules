import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import os
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class MoleculeClusterer:
    def __init__(self, smiles_file_path):
        self.smiles_file_path = smiles_file_path
        self.data = None
        self.fingerprints = []

    def load_data(self):
        self.data = pd.read_csv(self.smiles_file_path, sep='\t', usecols=['canonical_smiles'])

    @staticmethod
    def smiles_to_fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2) if mol else None

    @staticmethod
    def compute_fingerprint(smiles):
        return MoleculeClusterer.smiles_to_fingerprint(smiles)

    def parallel_generate_fingerprints(self):
        num_cpus = os.cpu_count()
        with Pool(num_cpus) as pool:
            self.fingerprints = list(tqdm(pool.imap(self.compute_fingerprint, self.data['canonical_smiles']), total=len(self.data)))
        self.fingerprints = [fp for fp in self.fingerprints if fp is not None]

    def calculate_similarity_matrix(self):
        num_fps = len(self.fingerprints)
        similarity_matrix = np.zeros((num_fps, num_fps))
        for i in tqdm(range(num_fps)):  # Adicionando barra de progresso aqui
            for j in range(num_fps):
                similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(self.fingerprints[i], self.fingerprints[j])
        return similarity_matrix

    def calculate_cosine_distance_matrix(self):
        fp_array = np.array([list(fp) for fp in self.fingerprints])
        cosine_sim_matrix = cosine_similarity(fp_array)
        cosine_dist_matrix = 1 - cosine_sim_matrix
        return cosine_dist_matrix

    def combine_similarity_matrices(self, tanimoto_matrix, cosine_matrix, alpha=0.5):
        combined_matrix = alpha * tanimoto_matrix + (1 - alpha) * cosine_matrix
        return combined_matrix

    def cluster_molecules(self, combined_similarity_matrix, threshold=0.8):
        clustering_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=1 - threshold)
        labels = clustering_model.fit_predict(1 - combined_similarity_matrix)
        self.data['Cluster'] = labels

    def save_clustered_data(self, output_file_path):
        self.data.to_csv(output_file_path, sep='\t', index=False)

    def generate_2d_visualization(self, threshold=0.8):
        num_fps = len(self.fingerprints)
        high_similarity_pairs = []

        for i in tqdm(range(num_fps)):
            for j in range(i + 1, num_fps):
                similarity = DataStructs.TanimotoSimilarity(self.fingerprints[i], self.fingerprints[j])
                if similarity >= threshold:
                    high_similarity_pairs.append((i, j, similarity))

        if not high_similarity_pairs:
            print("Nenhum par de moléculas com alta similaridade encontrado.")
            return

        # Preparando os dados para t-SNE
        indices = list(set([i for i, j, s in high_similarity_pairs] + [j for i, j, s in high_similarity_pairs]))
        embeddings = [self.fingerprints[i] for i in indices]
        embeddings = np.array([list(fp) for fp in embeddings])

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        # Definindo o número de clusters
        n_clusters = 5  # Este valor pode ser ajustado conforme necessário

        # Aplicar k-means nos resultados do t-SNE
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tsne_results)

        # Plotagem com cores para cada cluster
        plt.figure(figsize=(12, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, alpha=0.7, cmap='viridis')
        plt.title('Visualização 2D de Similaridade Molecular com Clusters (Tanimoto >= 0.8)')
        plt.xlabel('t-SNE Dimensão 1')
        plt.ylabel('t-SNE Dimensão 2')
        plt.colorbar(label='Cluster ID')
        plt.show()

    def save_state(self, file_path):
        state = {
            'data': self.data,
            'fingerprints': self.fingerprints
        }
        with open(file_path, 'wb') as file:
            pickle.dump(state, file)

    def load_state(self, file_path):
        with open(file_path, 'rb') as file:
            state = pickle.load(file)
            self.data = state['data']
            self.fingerprints = state['fingerprints']

def main():
    smiles_file_path = './nr_kinase_drug_info_kd_ki_manually_validated.tsv'
    output_file_path = './clustered_smiles.tsv'

    clusterer = MoleculeClusterer(smiles_file_path)
    # Tentar carregar o estado salvo anteriormente
    try:
        clusterer.load_state(state_file_path)
        print("Estado carregado com sucesso.")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        print("Nenhum estado salvo encontrado ou erro ao carregar o estado. Iniciando processamento do zero.")
        clusterer.load_data()
        clusterer.parallel_generate_fingerprints()

    if clusterer.fingerprints:
        # Gerar e visualizar a matriz de similaridade de Tanimoto
        clusterer.generate_2d_visualization(threshold=0.8)
        
        # Aqui, você pode adicionar a lógica de agrupamento, se necessário
        # tanimoto_matrix = clusterer.calculate_similarity_matrix()
        # clusterer.cluster_molecules(tanimoto_matrix, threshold=0.8)
        clusterer.save_clustered_data(output_file_path)
        
    else:
        print("Nenhum fingerprint válido foi encontrado.")

if __name__ == "__main__":
    main()


