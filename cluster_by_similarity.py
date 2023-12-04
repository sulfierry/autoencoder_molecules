import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from matplotlib import colormaps
import matplotlib.cm as cm
from rdkit.Chem import Descriptors

class MoleculeClusterer:
    def __init__(self, smiles_file_path):
        self.smiles_file_path = smiles_file_path
        self.data = None
        self.fingerprints = []

    def load_data(self):
        # Correção aqui: usar 'smiles_file_path' em vez de 'data_path'
        self.data = pd.read_csv(self.smiles_file_path, sep='\t')
        if 'pchembl_value' not in self.data.columns:
            raise ValueError("A coluna 'pchembl_value' não foi encontrada no arquivo.")

    def preprocess_data(self):
        # Classifica os valores de pchembl em grupos
        self.data['pchembl_group'] = self.data['pchembl_value'].apply(self.pchembl_group)
    
    def cluster_by_similarity(self, threshold=0.8):
        num_fps = len(self.fingerprints)
        clusters = []
        visited = set()

        for i in range(num_fps):
            if i in visited:
                continue

            cluster = [i]
            for j in range(i + 1, num_fps):
                similarity = DataStructs.TanimotoSimilarity(self.fingerprints[i], self.fingerprints[j])
                if similarity >= threshold:
                    cluster.append(j)
                    visited.add(j)

            clusters.append(cluster)

        return clusters
    
    @staticmethod
    def pchembl_group(value):
        # Lógica para classificar o valor pchembl em grupos
        if pd.isna(value):
            return 'sem_pchembl'
        elif 1 < value < 9:
            return 'grupo1'
        elif 9 <= value < 10:
            return 'grupo2'
        elif 10 <= value < 11:
            return 'grupo3'
        elif 11 <= value < 12:
            return 'grupo4'
        else:
            return '>12'

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

    def save_clustered_data(self, output_file_path):
        self.data.to_csv(output_file_path, sep='\t', index=False)

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


    def calculate_tsne(self):
        # Convertendo a lista de fingerprints para um array NumPy com uma barra de progresso do tqdm
        fingerprint_array = np.array([fp for fp in tqdm(self.fingerprints, desc='Processing Fingerprints') if fp is not None])
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(fingerprint_array)
        return tsne_results

    def plot_tsne(self, tsne_results):
        plt.figure(figsize=(12, 8))

        # Certifique-se de que os IDs dos clusters são inteiros
        cluster_ids = self.data['ClusterID'].astype(int)

        # Obter o número de clusters únicos
        num_clusters = len(set(cluster_ids))

        # Criar o mapa de cores
        cmap = plt.cm.viridis

        # Obter o tamanho de cada cluster
        cluster_sizes = self.data['ClusterID'].value_counts().sort_index()
        
        # Mapear cada cluster ID para o tamanho do cluster
        size_map = cluster_sizes.to_dict()
        cluster_colors = np.array([size_map[x] for x in cluster_ids])

        # Plotagem do TSNE com a coloração do cluster
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_colors, cmap=cmap, alpha=0.5)

        # Criar um objeto de barra de cores com um número inteiro de bins
        cbar = plt.colorbar(scatter, spacing='proportional', ticks=range(cluster_sizes.min(), cluster_sizes.max() + 1))
        cbar.set_label('Cluster Size')

        plt.title('Visualização 2D de Similaridade Molecular com Clusters')
        plt.xlabel('t-SNE Dimensão 1')
        plt.ylabel('t-SNE Dimensão 2')
        plt.show()
        
    def plot_cluster_size_distribution(self):
        # Contar o número de smiles em cada cluster
        cluster_sizes = self.data['ClusterID'].value_counts()

        # Criar um histograma dos tamanhos dos clusters
        plt.figure(figsize=(10, 6))
        plt.hist(cluster_sizes, bins=range(1, cluster_sizes.max() + 1), alpha=0.7, color='blue', edgecolor='black')

        plt.title('Distribuição do Número de Smiles por Cluster')
        plt.xlabel('Número de Smiles no Cluster')
        plt.ylabel('Contagem de Clusters')
        plt.grid(axis='y', alpha=0.75)

        # Mostrar média e mediana
        mean_size = cluster_sizes.mean()
        median_size = cluster_sizes.median()
        plt.axvline(mean_size, color='red', linestyle='dashed', linewidth=1, label=f'Média: {mean_size:.2f}')
        plt.axvline(median_size, color='green', linestyle='dashed', linewidth=1, label=f'Mediana: {median_size:.2f}')

        plt.legend()
        plt.show()

    def save_clusters_as_tsv(self, threshold=5):
        # Criar pasta 'clusters' se não existir
        os.makedirs('./clusters', exist_ok=True)

        # Colunas a serem incluídas no arquivo de saída
        output_columns = ['molregno', 'kinase_alvo', 'canonical_smiles', 'standard_value', 'standard_type', 'pchembl_value',
                        'nome_medicamento', 'ClusterID']

        # Calcula o peso molecular para cada SMILES e adiciona como uma nova coluna
        self.data['molecular_weight'] = self.data['canonical_smiles'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)) if x else None)

        for cluster_id in set(self.data['ClusterID']):
            cluster_data = self.data[self.data['ClusterID'] == cluster_id]

            # Ordenar os dados do cluster por peso molecular
            cluster_data_sorted = cluster_data.sort_values('molecular_weight')

            # Salvar apenas se o cluster contiver pelo menos 'threshold' SMILES
            if len(cluster_data_sorted) >= threshold:
                # Selecionar apenas as colunas desejadas para salvar
                cluster_data_to_save = cluster_data_sorted[output_columns]
                cluster_data_to_save.to_csv(f'./clusters/cluster_{cluster_id}.tsv', sep='\t', index=False)


def main():
    smiles_file_path = './nr_kinase_drug_info_kd_ki_manually_validated.tsv'
    output_file_path = './clustered_smiles.tsv'
    state_file_path = './molecule_clusterer_state.pkl'

    clusterer = MoleculeClusterer(smiles_file_path)

    try:
        clusterer.load_state(state_file_path)
        print("Estado carregado com sucesso.")
    except FileNotFoundError:
        print("Nenhum estado salvo encontrado. Iniciando processamento do zero.")
        clusterer.load_data()
        clusterer.preprocess_data()
        clusterer.parallel_generate_fingerprints()
        clusterer.save_state(state_file_path)
    except Exception as e:
        print(f"Erro ao carregar o estado: {e}")
        return

    if clusterer.fingerprints:
        tsne_results = clusterer.calculate_tsne()
        clusters = clusterer.cluster_by_similarity(threshold=0.8)
        clusterer.save_clusters_as_tsv(threshold=5)

        cluster_ids = [None] * len(clusterer.data)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                cluster_ids[idx] = cluster_id
        clusterer.data['ClusterID'] = cluster_ids

        clusterer.plot_tsne(tsne_results)
        clusterer.plot_cluster_size_distribution()

        clusterer.save_clustered_data(output_file_path)
        clusterer.save_state(state_file_path)
    else:
        print("Nenhum fingerprint válido foi encontrado.")

if __name__ == "__main__":
    main()
