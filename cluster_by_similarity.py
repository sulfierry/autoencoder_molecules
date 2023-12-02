import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import MinMaxScaler


def tanimoto_similarity(fp1, fp2):
    # Garantindo que fp1 e fp2 sejam objetos ExplicitBitVect
    if isinstance(fp1, np.ndarray):
        # Convertendo o NumPy array para string de bits e depois para ExplicitBitVect
        fp1_bits = ''.join(map(str, fp1.tolist()))
        fp1 = DataStructs.ExplicitBitVect(len(fp1_bits))
        for i, bit in enumerate(fp1_bits):
            if bit == '1':
                fp1.SetBit(i)

    if isinstance(fp2, np.ndarray):
        # Convertendo o NumPy array para string de bits e depois para ExplicitBitVect
        fp2_bits = ''.join(map(str, fp2.tolist()))
        fp2 = DataStructs.ExplicitBitVect(len(fp2_bits))
        for i, bit in enumerate(fp2_bits):
            if bit == '1':
                fp2.SetBit(i)

    return DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)


class TSNEClusterer:
    def __init__(self, data_path, pkidb_path):
        self.data_path = data_path
        self.pkidb_path = pkidb_path
        self.data = None
        self.pkidb_data = None
        self.tsne_results = []
        self.group_labels = []
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path, sep='\t')
        self.pkidb_data = pd.read_csv(self.pkidb_path, sep='\t', usecols=['Canonical_Smiles'])
        print(f'Dados carregados: {self.data.shape[0]} linhas de "data", {self.pkidb_data.shape[0]} linhas de "pkidb_data".')

        
    def smiles_to_fingerprint(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Retornando diretamente o ExplicitBitVect, sem convertê-lo para uma string
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
        except Exception as e:
            print(f"Erro ao converter SMILES: {smiles} - {e}")
        return None

    
    def preprocess_data(self):
        self.pkidb_data['fingerprint'] = self.pkidb_data['Canonical_Smiles'].apply(self.smiles_to_fingerprint)
        failed_conversions = self.pkidb_data['fingerprint'].isnull().sum()
        print(f'Fingerprints convertidos com {failed_conversions} falhas.')
        self.pkidb_data.dropna(subset=['fingerprint'], inplace=True)

        
    def calculate_similarity_matrix(self):
        valid_fingerprints = [fp for fp in self.pkidb_data['fingerprint'] if fp is not None]
        print(f'Calculando matriz de similaridade para {len(valid_fingerprints)} fingerprints válidos...')
        if len(valid_fingerprints) < 2:
            print("Não há fingerprints suficientes para calcular a matriz de similaridade.")
            return None
    
        # Retorna a matriz de distância condensada diretamente
        similarity_matrix = pdist(valid_fingerprints, lambda u, v: 1 - tanimoto_similarity(u, v))
        print(f'Matriz de similaridade calculada com tamanho: {similarity_matrix.size}.')
        return similarity_matrix
    
    def normalize_tsne_results(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.tsne_results = scaler.fit_transform(self.tsne_results)
        print(f'Resultados do t-SNE normalizados.')
        
    def cluster_molecules(self, condensed_similarity_matrix, threshold=0.8):
        print(f'Agrupando com base na matriz de similaridade condensada...')
        linked = linkage(condensed_similarity_matrix, 'single')
        clusters = fcluster(linked, t=threshold, criterion='distance')
        print(f'{len(np.unique(clusters))} clusters formados.')
        return clusters


    def calculate_tsne(self):
        similarity_matrix = self.calculate_similarity_matrix()

        if similarity_matrix is None:
            return  # Encerrar a função se a matriz de similaridade não puder ser calculada

        self.pkidb_data['cluster'] = self.cluster_molecules(similarity_matrix)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for cluster_id in self.pkidb_data['cluster'].unique():
                cluster_data = self.pkidb_data[self.pkidb_data['cluster'] == cluster_id]
                fingerprints = list(cluster_data['fingerprint'])
                futures.append(executor.submit(self.calculate_tsne_for_fingerprints, fingerprints, cluster_id))

            for future in concurrent.futures.as_completed(futures):
                result, cluster_id = future.result()
                if result is not None:
                    self.tsne_results.extend(result)
                    self.group_labels.extend([cluster_id] * len(result))

    def calculate_tsne_for_fingerprints(self, fingerprints, cluster_id):
        if len(fingerprints) > 5:
            fingerprints_matrix = np.array(fingerprints)
            tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(fingerprints_matrix) - 1))
            tsne_result = tsne.fit_transform(fingerprints_matrix)
            return tsne_result, cluster_id
        return [], cluster_id
        
    def save_tsne_results(self, filename):
        tsne_df = pd.DataFrame(self.tsne_results, columns=['x', 'y'])
        tsne_df['cluster'] = self.group_labels
        tsne_df.to_csv(filename, sep='\t', index=False)
        print(f'Resultados do t-SNE salvos em {filename}.')

    # Antes de chamar plot_tsne, verifique os clusters
    print("Clusters únicos:", np.unique(self.group_labels))
    
    def plot_tsne(self):
        tsne_df = pd.DataFrame(self.tsne_results, columns=['x', 'y'])
        tsne_df['cluster'] = self.group_labels
        
        # Verifique se os clusters variam como esperado
        print(tsne_df['cluster'].value_counts())
    
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(tsne_df['x'], tsne_df['y'], c=tsne_df['cluster'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
        plt.title('t-SNE Clustering Based on Tanimoto Similarity')
        plt.xlabel('t-SNE feature 0')
        plt.ylabel('t-SNE feature 1')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()



    def run(self):
        self.load_data()
        self.preprocess_data()
        self.calculate_tsne()
        self.normalize_tsne_results()
        self.plot_tsne()
        self.save_tsne_results('./tsne_cluster_similarity.tsv')  # Especifica o caminho desejado para o arquivo de saída
    


def main():
    tsne_clusterer = TSNEClusterer('./nr_kinase_drug_info_kd_ki_manually_validated.tsv', './pkidb_2023-06-30.tsv')
    tsne_clusterer.run()

if __name__ == '__main__':
    main()

