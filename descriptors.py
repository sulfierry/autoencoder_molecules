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
    for name in descriptor_names:
        plt.figure()
        plt.hist(descriptor_data[name], bins=30, edgecolor='black')
        plt.title(f'Histograma do Descritor: {name}')
        plt.xlabel(name)
        plt.ylabel('Frequência')
        plt.show()

# Altere para o caminho do seu arquivo
file_path = './nr_kinase_drug_info_kd_ki_manually_validated.tsv'

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

# Plotando histogramas
plot_histograms(descriptor_data, descriptor_names)
