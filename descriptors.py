import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return {
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NRB': Descriptors.NumRotatableBonds(mol)
            }
    except Exception as e:
        print(f"Erro ao processar SMILES: {smiles}: {e}")
    return None

def plot_histograms(descriptor_data, descriptor_names):
    fig, axs = plt.subplots(3, 2, figsize=(10, 15)) # 3 linhas, 2 colunas
    axs = axs.flatten() # Transformar o array 2D de eixos em 1D para facilitar o acesso

    for i, name in enumerate(descriptor_names):
        axs[i].hist(descriptor_data[name], bins=30, edgecolor='black')
        axs[i].set_title(f'Histograma do Descritor: {name}')
        axs[i].set_xlabel(name)
        axs[i].set_ylabel('Frequência')

    plt.tight_layout() # Ajusta o layout para que não haja sobreposição
#    plt.savefig('./nr_descriptros_histogram.png')
    plt.show()

# Altere para o caminho do seu arquivo
file_path = 'nr_kinase_drug_info_kd_ki_manually_validated.tsv'

# Lendo o arquivo
data = pd.read_csv(file_path, sep='\t')

# Calculando descritores
descriptor_names = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'NRB']
descriptor_data = {name: [] for name in descriptor_names}

for smiles in data['canonical_smiles']:
    descriptors = calculate_descriptors(smiles)
    if descriptors:
        for name in descriptor_names:
            descriptor_data[name].append(descriptors[name])

# Convertendo os dados dos descritores para um DataFrame e salvando em um arquivo .tsv
descriptor_df = pd.DataFrame(descriptor_data)
descriptor_df.to_csv('descritores_moleculares.tsv', sep='\t', index=False)

# Plotando histogramas
plot_histograms(descriptor_data, descriptor_names)


import pandas as pd
import matplotlib.pyplot as plt

# Altere para o caminho do seu arquivo pkidb
file_path_pkidb = './pkidb_2023-06-30.tsv'

# Lendo os dados do pkidb
data_pkidb = pd.read_csv(file_path_pkidb, sep='\t')

# Nomes dos descritores
descritores = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'NRB']

# Lendo os dados dos descritores calculados anteriormente
# Altere para o caminho do seu arquivo de descritores calculados
file_path_descritores_calculados = 'descritores_moleculares.tsv'
data_calculados = pd.read_csv(file_path_descritores_calculados, sep='\t')

# Função para plotar os histogramas normalizados e sobrepostos
def plot_histograms_normalizados_sobrepostos(data1, data2, descritores):
    fig, axs = plt.subplots(3, 2, figsize=(10, 15)) # 3 linhas, 2 colunas
    axs = axs.flatten() # Transformar o array 2D de eixos em 1D

    for i, desc in enumerate(descritores):
        axs[i].hist(data1[desc], bins=30, alpha=0.5, label='nr_Chembl', edgecolor='black', density=True)
        axs[i].hist(data2[desc], bins=30, alpha=0.5, label='PKIDB', edgecolor='black', density=True)
        axs[i].set_title(f'{desc}')
        #axs[i].set_xlabel(desc)
        axs[i].set_ylabel('Density')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig('./pkidb_nr_chembl_histogram.png')
    plt.show()

# Plotando os histogramas normalizados e sobrepostos
plot_histograms_normalizados_sobrepostos(data_calculados, data_pkidb, descritores)
