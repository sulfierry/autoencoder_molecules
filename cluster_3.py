import os
import numpy as np
import pandas as pd
from rdkit import Chem
import concurrent.futures
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor



class TSNEClusterer:
    def __init__(self, data_path, pkidb_path):
        self.data_path = data_path
        self.pkidb_path = pkidb_path
        self.data = None
        self.pkidb_data = None
        self.tsne_results = []  # Inicializa a lista aqui
        self.group_labels = []  # Inicializa a lista aq

    def save_tsne_results(self, file_path):
        tsne_df = pd.DataFrame(self.tsne_results, columns=['x', 'y'])
        tsne_df['group'] = self.group_labels
        tsne_df.to_csv(file_path, sep='\t', index=False)
        
    def load_data(self):
        self.data = pd.read_csv(self.data_path, sep='\t')
        self.pkidb_data = pd.read_csv(self.pkidb_path, sep='\t', usecols=['Canonical_Smiles'])

    @staticmethod
    def smiles_to_fingerprint(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol and mol.GetNumAtoms() > 0 and Chem.SanitizeMol(mol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
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
            
    def calculate_tsne_for_group(self, group_data):
        # Este método será chamado em paralelo para cada grupo
        fingerprints = [self.smiles_to_fingerprint(smiles) for smiles in group_data['canonical_smiles'] if smiles]
        fingerprints_matrix = np.array([fp for fp in fingerprints if fp is not None])
        if len(fingerprints_matrix) > 5:
            tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(fingerprints_matrix) - 1))
            tsne_result = tsne.fit_transform(fingerprints_matrix)
            return tsne_result, group_data['pchembl_group'].iloc[0]

    @staticmethod
    def smiles_to_fingerprint(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
        except Exception as e:
            print(f"Erro ao converter SMILES: {smiles} - {e}")
        return None

    @staticmethod
    def process_group_data(smiles_list):
        fingerprints = [TSNEClusterer.smiles_to_fingerprint(smiles) for smiles in smiles_list if smiles]
        fingerprints_matrix = np.array([fp for fp in fingerprints if fp is not None])
        if len(fingerprints_matrix) > 5:
            tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(fingerprints_matrix) - 1))
            tsne_result = tsne.fit_transform(fingerprints_matrix)
            return tsne_result
        return None
            
    def calculate_tsne(self):
        # Processamento paralelo para calcular t-SNE para cada grupo de pChEMBL
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for group in self.data['pchembl_group'].unique():
                group_data = self.data[self.data['pchembl_group'] == group]
                futures.append(executor.submit(self.calculate_tsne_for_group, group_data))
    
            for future in concurrent.futures.as_completed(futures):
                result, group = future.result()
                if result is not None:
                    self.tsne_results.extend(result)
                    self.group_labels.extend([group] * len(result))
    
        # Adicionar os resultados t-SNE ao DataFrame
        self.tsne_df = pd.DataFrame(self.tsne_results, columns=['x', 'y'])
        self.tsne_df['group'] = self.group_labels
    
        # Calcular t-SNE para dados PKIDB
        pkidb_fingerprints = [fp for fp in self.pkidb_data['fingerprint'] if fp is not None]
        if len(pkidb_fingerprints) > 0:
            tsne_pkidb = TSNE(n_components=2, random_state=0, perplexity=30)
            pkidb_fingerprints_matrix = np.array(pkidb_fingerprints)
            tsne_results_pkidb = tsne_pkidb.fit_transform(pkidb_fingerprints_matrix)
    
            # Atualizar o DataFrame pkidb_data com os resultados t-SNE
            self.pkidb_data['x'], self.pkidb_data['y'] = tsne_results_pkidb[:, 0], tsne_results_pkidb[:, 1]
            self.pkidb_data['group'] = 'PKIDB Ligantes'
    
            # Adicionar os dados do PKIDB ao DataFrame tsne_df
            self.tsne_df = pd.concat([self.tsne_df, self.pkidb_data[['x', 'y', 'group']]], ignore_index=True)
        else:
            print("Nenhum fingerprint válido encontrado para PKIDB.")


    def calculate_tsne_for_group(self, group_data):
        try:
            fingerprints = [self.smiles_to_fingerprint(smiles) for smiles in group_data['canonical_smiles'] if smiles]
            fingerprints_matrix = np.array([fp for fp in fingerprints if fp is not None])
            if len(fingerprints_matrix) > 5:
                tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(fingerprints_matrix) - 1))
                tsne_result = tsne.fit_transform(fingerprints_matrix)
                return tsne_result, group_data['pchembl_group'].iloc[0]
        except Exception as e:
            print(f"Erro no cálculo do t-SNE para o grupo: {group_data['pchembl_group'].iloc[0]} - {e}")
        return [], group_data['pchembl_group'].iloc[0]  # Retorna lista vazia e grupo

    def calculate_tsne_pkidb(self):
        # Certifique-se de que os fingerprints do PKIDB estão sendo calculados corretamente
        self.pkidb_data['fingerprint'] = self.pkidb_data['Canonical_Smiles'].apply(self.smiles_to_fingerprint)
        self.pkidb_data.dropna(subset=['fingerprint'], inplace=True)
    
        # Calculando t-SNE para o PKIDB
        if not self.pkidb_data.empty:
            fingerprints = list(self.pkidb_data['fingerprint'])
            fingerprints_matrix = np.array(fingerprints)
            tsne_pkidb = TSNE(n_components=2, random_state=0, perplexity=30)
            tsne_results_pkidb = tsne_pkidb.fit_transform(fingerprints_matrix)
        
            # Salvando os resultados
             
            tsne_pkidb_df = pd.DataFrame(tsne_results_pkidb, columns=['x', 'y'])
            tsne_pkidb_df.to_csv('PKIDB_tSNE_results.tsv', sep='\t', index=False)
        else:
             print("Nenhum fingerprint válido encontrado para PKIDB.")

    def plot_tsne(self):
        plt.figure(figsize=(12, 8))
        
        # Cores e ordem de plotagem
        colors = {
            'sem_pchembl': 'grey',
            'grupo1': 'blue',
            'grupo2': 'green',
            'grupo3': 'yellow',
            'grupo4': 'orange',
            'grupo5': 'purple',
            'PKIDB Ligantes': 'red'
        }
        plot_order = [
            'sem_pchembl', 'grupo1', 'grupo2', 'grupo3',
            'grupo4', 'grupo5', 'PKIDB Ligantes'
        ]
    
        # Plotagem de acordo com a ordem definida
        for group in plot_order[:-1]:  # Exclua 'PKIDB Ligantes' desta iteração
            if group in self.tsne_df['group'].unique():
                subset = self.tsne_df[self.tsne_df['group'] == group]
                plt.scatter(subset['x'], subset['y'], color=colors[group], label=group, alpha=0.5)
        
        # Verificar se as colunas 'x' e 'y' existem no DataFrame pkidb_data
        if 'x' in self.pkidb_data and 'y' in self.pkidb_data:
            pkidb_subset = self.pkidb_data
            plt.scatter(pkidb_subset['x'], pkidb_subset['y'], color=colors['PKIDB Ligantes'], label='PKIDB Ligantes', alpha=0.6)
        else:
            print("Colunas 'x' e 'y' não encontradas no DataFrame pkidb_data.")
    
        plt.legend()
        plt.title('Distribuição dos Ligantes por Grupo de pChEMBL Value com PKIDB (2D)')
        plt.xlabel('t-SNE feature 0')
        plt.ylabel('t-SNE feature 1')
        plt.savefig('./tsne_chembl_pkidb_clusters.png')
        plt.show()
        

    def save_data(self):
        # Salvar grupos sem pchembl_value
        self.data[self.data['pchembl_group'] == 'sem_pchembl'].to_csv('./kinases_sem_pchembl_value.tsv', sep='\t', index=False)
    
        # Salvar dados t-SNE de pChEMBL
        for group in self.data['pchembl_group'].unique():
            group_data = self.data[self.data['pchembl_group'] == group]
            group_data.to_csv(f'./kinases_{group}.tsv', sep='\t', index=False)
    
        # Salvar dados t-SNE de PKIDB
        self.pkidb_data.to_csv('./pkidb_tsne_results.tsv', sep='\t', index=False)

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.calculate_tsne()
        self.calculate_tsne_pkidb() 
        self.plot_tsne()
        self.save_data()

      
def calculate_tsne_parallel(tsne_clusterer, group):
    group_data = tsne_clusterer.data[tsne_clusterer.data['pchembl_group'] == group]
    return tsne_clusterer.calculate_tsne_for_group(group_data)

def plot_normalized_tsne(chembl_file, pkidb_file):
    chembl_data = pd.read_csv(chembl_file, sep='\t')
    pkidb_data = pd.read_csv(pkidb_file, sep='\t')

    # Normalizar os dados
    scaler = MinMaxScaler()
    chembl_data[['x', 'y']] = scaler.fit_transform(chembl_data[['x', 'y']])
    pkidb_data[['x', 'y']] = scaler.transform(pkidb_data[['x', 'y']])

    # Cores e ordem de plotagem
    colors = {
        'sem_pchembl': 'grey',
        'grupo1': 'blue',
        'grupo2': 'green',
        'grupo3': 'yellow',
        'grupo4': 'orange',
        'grupo5': 'purple',
        'PKIDB Ligantes': 'red'
    }
    plot_order = [
        'sem_pchembl', 'grupo1', 'grupo2', 'grupo3',
        'grupo4', 'grupo5', 'PKIDB Ligantes'
    ]

    # Plotar os dados normalizados com cores específicas para cada grupo
    plt.figure(figsize=(12, 8))
    
    # Plotagem de acordo com a ordem definida
    for group in plot_order[:-1]:  # Exclua 'PKIDB Ligantes' desta iteração
        if group in chembl_data['group'].unique():
            subset = chembl_data[chembl_data['group'] == group]
            plt.scatter(subset['x'], subset['y'], color=colors[group], label=group, alpha=0.5)

    # Plotar os dados do PKIDB
    if 'x' in pkidb_data.columns and 'y' in pkidb_data.columns:
        plt.scatter(pkidb_data['x'], pkidb_data['y'], color=colors['PKIDB Ligantes'], label='PKIDB Ligantes', alpha=0.5)

    plt.legend()
    plt.title('Normalized t-SNE Plot of Chembl and PKIDB')
    plt.xlabel('t-SNE feature 0')
    plt.ylabel('t-SNE feature 1')
    plt.show()

def main():
    tsne_clusterer = TSNEClusterer('./nr_kinase_drug_info_kd_ki_manually_validated.tsv', './pkidb_2023-06-30.tsv')
    tsne_clusterer.load_data()
    tsne_clusterer.preprocess_data()
    
    # Utilizar todas as CPUs disponíveis para calcular t-SNE em paralelo para cada grupo
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(calculate_tsne_parallel, tsne_clusterer, group): group for group in tsne_clusterer.data['pchembl_group'].unique()}
        
        for future in concurrent.futures.as_completed(futures):
            result, group = future.result()
            if result is not None:
                tsne_clusterer.tsne_results.extend(result)
                tsne_clusterer.group_labels.extend([group] * len(result))
    
    tsne_clusterer.tsne_df = pd.DataFrame(tsne_clusterer.tsne_results, columns=['x', 'y'])
    tsne_clusterer.tsne_df['group'] = tsne_clusterer.group_labels
    
    # Salvar os resultados do t-SNE para os dados do ChEMBL
    tsne_clusterer.save_tsne_results('chembl_tsne_results.tsv')

    # Calcular t-SNE para os dados do PKIDB e salvar os resultados
    tsne_clusterer.calculate_tsne_pkidb()
    tsne_clusterer.pkidb_data.to_csv('pkidb_tsne_results.tsv', sep='\t', index=False)

    # Plotar os dados normalizados
    plot_normalized_tsne('chembl_tsne_results.tsv', 'PKIDB_tSNE_results.tsv')

    # Continuar com o restante da execução
    # tsne_clusterer.plot_tsne()
    # tsne_clusterer.save_data()

if __name__ == '__main__':
    main()
