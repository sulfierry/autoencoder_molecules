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
        
        # Inicialize similarity_matrix como None para garantir que ela tenha um valor em todos os casos
        similarity_matrix = None
        
        if len(valid_fingerprints) < 2:
            print("Não há fingerprints suficientes para calcular a matriz de similaridade.")
        else:
            # Calcula a matriz de distância condensada
            similarity_matrix = pdist(valid_fingerprints, lambda u, v: 1 - tanimoto_similarity(u, v))
            print(f'Matriz de similaridade calculada com tamanho: {similarity_matrix.size}.')
        
        # A visualização da distribuição dos valores de similaridade só deve ocorrer se similarity_matrix não for None
        if similarity_matrix is not None:
            print("Valores de similaridade:", np.unique(similarity_matrix))
            plt.hist(similarity_matrix, bins=50)
            plt.title('Distribuição dos Valores de Similaridade')
            plt.xlabel('Similaridade de Tanimoto')
            plt.ylabel('Frequência')
            plt.show()

            # Visualize o dendrograma
            plt.figure(figsize=(10, 7))
            plt.title("Dendrograma dos Clusters")
            dendrogram(linkage(similarity_matrix, method='ward'))
            plt.show()
        return similarity_matrix
        
        return similarity_matrix

    
    def normalize_tsne_results(self):
        print(f'Resultados do t-SNE antes da normalização: {self.tsne_results[:5]}')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.tsne_results = scaler.fit_transform(self.tsne_results)
        print(f'Resultados do t-SNE após a normalização: {self.tsne_results[:5]}')
        print(f'Resultados do t-SNE normalizados.')
        
    def cluster_molecules(self, condensed_similarity_matrix, threshold=0.2):  # Ajuste o valor do limiar
        print(f'Agrupando com base na matriz de similaridade condensada com limiar {threshold}...')
        linked = linkage(condensed_similarity_matrix, method='ward')  # Experimente com diferentes métodos
        clusters = fcluster(linked, t=threshold, criterion='distance')
        print(f'Clusters formados: {np.unique(clusters)}')
        print(f'Contagem por cluster: {np.bincount(clusters)[1:]}')
        return clusters
        
    def calculate_tsne(self):
        similarity_matrix = self.calculate_similarity_matrix()
        if similarity_matrix is None:
            print("Não foi possível calcular a matriz de similaridade.")
            return

        clusters = self.cluster_molecules(similarity_matrix, threshold=0.5)
        self.pkidb_data['cluster'] = clusters

        # Verifique se o número de clusters é maior que 1 para continuar com o t-SNE
        if len(np.unique(clusters)) <= 1:
            print("Número insuficiente de clusters para calcular t-SNE.")
            return

        # Processamento paralelo para cálculo do t-SNE
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for cluster_id in np.unique(clusters):
                cluster_data = self.pkidb_data[self.pkidb_data['cluster'] == cluster_id]
                fingerprints = list(cluster_data['fingerprint'])
                if fingerprints:  # Verifique se a lista de fingerprints não está vazia
                    futures.append(executor.submit(self.calculate_tsne_for_fingerprints, fingerprints, cluster_id))

            for future in concurrent.futures.as_completed(futures):
                result, cluster_id = future.result()
                if result:  # Verifique se o resultado não está vazio
                    self.tsne_results.extend(result)
                    self.group_labels.extend([cluster_id] * len(result))

        # Verifique se os resultados do t-SNE não estão vazios antes de normalizar
        if not self.tsne_results:
            print("Nenhum resultado do t-SNE para normalizar.")
            return

        self.normalize_tsne_results()
        self.plot_tsne()
        self.save_tsne_results('./tsne_cluster_similarity.tsv')
    
    def calculate_tsne_for_fingerprints(self, fingerprints, cluster_id):
        # Certifique-se de que há fingerprints suficientes para calcular o t-SNE
        if len(fingerprints) > 5:
            try:
                fingerprints_matrix = np.array(fingerprints)
                tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(fingerprints_matrix) - 1))
                tsne_result = tsne.fit_transform(fingerprints_matrix)
                return tsne_result, cluster_id
            except Exception as e:
                print(f"Erro ao calcular t-SNE para o cluster {cluster_id}: {e}")
        else:
            print(f"Não há fingerprints suficientes para o cluster {cluster_id}.")
        return [], cluster_id  # Retorna uma lista vazia se não houver dados suficientes ou ocorrer um erro

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

    
    def plot_tsne(self):
        print("Clusters únicos:", np.unique(self.group_labels))

        tsne_df = pd.DataFrame(self.tsne_results, columns=['x', 'y'])
        tsne_df['cluster'] = self.group_labels
        
        # Verificação final dos clusters antes da plotagem
        print(f'Clusters finais a serem plotados: {np.unique(tsne_df["cluster"])}')
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
        
        if self.tsne_results:  # Verifique se há resultados antes de normalizar
            self.normalize_tsne_results()
            self.plot_tsne()
            self.save_tsne_results('./tsne_cluster_similarity.tsv')
        else:
            print("Não há resultados do t-SNE para processar.")
    


def main():
    tsne_clusterer = TSNEClusterer('./nr_kinase_drug_info_kd_ki_manually_validated.tsv', './pkidb_2023-06-30.tsv')
    tsne_clusterer.run()

if __name__ == '__main__':
    main()

