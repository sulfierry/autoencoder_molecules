import os
import numpy as np
import pandas as pd
from rdkit import Chem
import concurrent.futures
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor


class TSNEClusterer:
    def __init__(self, data_path, pkidb_path):
        self.data_path = data_path
        self.pkidb_path = pkidb_path
        self.data = None
        self.pkidb_data = None
        self.tsne_df = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path, sep='\t')
        self.pkidb_data = pd.read_csv(self.pkidb_path, sep='\t', usecols=['Canonical_Smiles'])

    def smiles_to_fingerprint(self, smiles):
        # Utilize a função existente ou melhore para tratar exceções e moléculas inválidas
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
        except Exception as e:
            print(f"Erro ao converter SMILES: {smiles} - {e}")
        return None

    def preprocess_data(self):
        self.data['pchembl_group'] = self.data['pchembl_value'].apply(self.pchembl_group)
        self.pkidb_data['fingerprint'] = self.pkidb_data['Canonical_Smiles'].apply(self.smiles_to_fingerprint)
        self.pkidb_data.dropna(subset=['fingerprint'], inplace=True)
      
    
    def pchembl_group(self, value):
        if pd.isna(value):
            return 'sem_pchembl'
        elif 1 < value < 8:
            return 'grupo1_(1 - 8)'
        elif 8 <= value < 9:
            return 'grupo2_(8 - 9)'
        elif 9 <= value < 10:
            return 'grupo3_(9 - 10)'
        elif 10 <= value < 11:
            return 'grupo4_(10 - 11)'
        elif 11 <= value < 12:
            return 'grupo5_(11 - 12)'
        else:
            return '>12'

    def calculate_tsne_for_group(self, group_data):
        # Este método será chamado em paralelo para cada grupo
        fingerprints = [self.smiles_to_fingerprint(smiles) for smiles in group_data['smiles'] if smiles]
        fingerprints_matrix = np.array([fp for fp in fingerprints if fp is not None])
        if len(fingerprints_matrix) > 5:
            tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(fingerprints_matrix) - 1))
            tsne_result = tsne.fit_transform(fingerprints_matrix)
            return tsne_result, group_data['pchembl_group'].iloc[0]
    
    
    def calculate_tsne(self):
        # Preparar dados para t-SNE e plotagem
        tsne_results = []
        group_labels = []
    
        # Defina a função que será executada em paralelo
        def process_group_data(group_data):
            fingerprints = [self.smiles_to_fingerprint(smiles) for smiles in group_data['smiles'] if smiles]
            fingerprints_matrix = np.array([fp for fp in fingerprints if fp is not None])
            if len(fingerprints_matrix) > 5:
                tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(fingerprints_matrix) - 1))
                tsne_result = tsne.fit_transform(fingerprints_matrix)
                return tsne_result, group_data['pchembl_group'].iloc[0]
            return None, None
    
        # Usar ProcessPoolExecutor para processar dados em paralelo
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_group_data, self.data[self.data['pchembl_group'] == group])
                       for group in self.data['pchembl_group'].unique()]
    
            for future in concurrent.futures.as_completed(futures):
                result, group = future.result()
                if result is not None:
                    tsne_results.extend(result)
                    group_labels.extend([group] * len(result))
    
        # Adicione os resultados à instância da classe
        self.tsne_df = pd.DataFrame(tsne_results, columns=['x', 'y'])
        self.tsne_df['group'] = group_labels

        
    def plot_tsne(self):
        plt.figure(figsize=(12, 8))
    
        # Ordem desejada para a plotagem
        plot_order = [
            'sem_pchembl', 
            'grupo1_(1 - 8)', 
            'grupo2_(8 - 9)', 
            'grupo3_(9 - 10)', 
            'grupo4_(10 - 11)', 
            'grupo5_(11 - 12)',
            'PKIDB Ligantes'  # Adicione 'PKIDB Ligantes' como o último a ser plotado
        ]
    
        # Cores específicas para cada grupo
        colors = {
            'sem_pchembl': 'grey',
            'grupo1_(1 - 8)': 'blue',
            'grupo2_(8 - 9)': 'green',
            'grupo3_(9 - 10)': 'yellow',
            'grupo4_(10 - 11)': 'orange',
            'grupo5_(11 - 12)': 'purple',
            'PKIDB Ligantes': 'red'  # Cor inconfundível para PKIDB
        }
    
        # Plotagem de acordo com a ordem definida
        for group in plot_order[:-1]:  # Exclua 'PKIDB Ligantes' desta iteração
            if group in self.tsne_df['group'].unique():
                subset = self.tsne_df[self.tsne_df['group'] == group]
                plt.scatter(subset['x'], subset['y'], color=colors[group], label=group, alpha=0.5)
    
        # Garantir que PKIDB seja plotado por último e seja claramente visível
        if 'PKIDB Ligantes' in plot_order:
            pkidb_subset = self.pkidb_data
            plt.scatter(pkidb_subset['x'], pkidb_subset['y'], color=colors['PKIDB Ligantes'], label='PKIDB Ligantes', alpha=0.6)
    
        plt.legend()
        plt.title('Distribuição dos Ligantes por Grupo de pChEMBL Value com PKIDB (2D)')
        plt.xlabel('t-SNE feature 0')
        plt.ylabel('t-SNE feature 1')
        plt.show()
        plt.savefig('./tsne_chembl_pkidb_clusters.png')
        
    def save_data(self):
        # Salvar grupos sem pchembl_value
        self.data[self.data['pchembl_group'] == 'sem_pchembl'].to_csv('./kinases_sem_pchembl_value.tsv', sep='\t', index=False)
    
        # Salvar dados t-SNE de pChEMBL
        for group in self.data['pchembl_group'].unique():
            group_data = self.data[self.data['pchembl_group'] == group]
            group_data.to_csv(f'./kinases_{group}.tsv', sep='\t', index=False)
    
        # Salvar dados t-SNE de PKIDB
        self.pkidb_data.to_csv('./PKIDB_tSNE_results.tsv', sep='\t', index=False)

  
    def run(self):
        self.load_data()
        self.preprocess_data()
        self.calculate_tsne()
        self.plot_tsne()
        self.save_data()

def main():
    tsne_clusterer = TSNEClusterer('./kinase_ligands_pchembl_Value.tsv', '../PKIDB/pkidb_2023-06-30.tsv')
    tsne_clusterer.run()

if __name__ == '__main__':
    main()
