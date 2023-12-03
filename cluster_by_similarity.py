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

    @staticmethod
    def pchembl_group(value):
        # Lógica para classificar o valor pchembl em grupos
        if pd.isna(value):
            return 'sem_pchembl'
        elif 1 < value < 8:
            return 'grupo1'
        elif 8 <= value < 9:
            return 'grupo2'
        elif 9 <= value < 10:
            return 'grupo3'
        elif 10 <= value < 11:
            return 'grupo4'
        elif 11 <= value < 12:
            return 'grupo5'
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

    def calculate_similarity_matrix(self):
        num_fps = len(self.fingerprints)
        similarity_matrix = np.zeros((num_fps, num_fps))
        for i in tqdm(range(num_fps)):  # Adicionando barra de progresso aqui
            for j in range(num_fps):
                similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(self.fingerprints[i], self.fingerprints[j])
        return similarity_matrix

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


    def plot_tsne(self, tsne_results, group_labels):
        plt.figure(figsize=(12, 8))

        # Cores e ordem de plotagem baseada nos grupos pchembl
        group_to_color = {
            'sem_pchembl': 'grey',
            'grupo1': 'blue',
            'grupo2': 'green',
            'grupo3': 'yellow',
            'grupo4': 'orange',
            'grupo5': 'purple',
            '>12': 'red'
        }

        for group, color in group_to_color.items():
            # Filtrar os resultados do t-SNE pelo grupo atual
            indices = [i for i, g in enumerate(group_labels) if g == group]
            subset = tsne_results[indices, :]
            plt.scatter(subset[:, 0], subset[:, 1], color=color, label=group, alpha=0.5)

        plt.legend()
        plt.title('Visualização 2D de Similaridade Molecular com Clusters')
        plt.xlabel('t-SNE Dimensão 1')
        plt.ylabel('t-SNE Dimensão 2')
        plt.show()


def main():
    # Caminhos dos arquivos
    smiles_file_path = './nr_kinase_drug_info_kd_ki_manually_validated.tsv'
    output_file_path = './clustered_smiles.tsv'
    state_file_path = './molecule_clusterer_state.pkl'

    # Instanciar a classe MoleculeClusterer
    clusterer = MoleculeClusterer(smiles_file_path)

    # Tentar carregar o estado salvo anteriormente
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

    # Realizar cálculos de similaridade e visualização com base nos grupos de pchembl_value
    if clusterer.fingerprints:

        # TSNE calculation
        tsne_results = clusterer.calculate_tsne()

        # Get group labels for plotting (ensure this matches with the tsne_results)
        group_labels = clusterer.data['pchembl_group'].values

        # Visualização dos resultados com cores baseadas nos grupos pchembl
        clusterer.plot_tsne(tsne_results, group_labels)
        
        # Salvar os dados clusterizados
        clusterer.save_clustered_data(output_file_path)

        # Salvar o estado final
        clusterer.save_state(state_file_path)

    else:
        print("Nenhum fingerprint válido foi encontrado.")

if __name__ == "__main__":
    main()
