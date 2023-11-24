import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Carregar dados
file_path = './kinase_ligands_pchembl_Value.tsv'
data = pd.read_csv(file_path, sep='\t')


# Função para converter SMILES em fingerprint
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) if mol else None

# Função para agrupar por 'pchembl_value'
def pchembl_group(value):
    if pd.isna(value):
        return 'sem_pchembl'
    elif 1 < value < 4:
        return 'grupo1'
    elif 4 <= value < 6:
        return 'grupo2'
    elif 6 <= value < 8:
        return 'grupo3'
    elif 8 <= value <= 10:
        return 'grupo4'
    else:
        return 'outros'

# Aplicar agrupamento
data['pchembl_group'] = data['pchembl_value'].apply(pchembl_group)


# Filtrar e salvar kinases sem 'pchembl_value'
kinases_sem_pchembl = data[data['pchembl_group'] == 'sem_pchembl']
kinases_sem_pchembl.to_csv('./kinases_sem_pchembl_value.tsv', sep='\t', index=False)

# Remover kinases sem 'pchembl_value' para análise t-SNE
data = data[data['pchembl_group'] != 'sem_pchembl']

# Preparar dados para t-SNE e plotagem
tsne_results = []
group_labels = []



for group in data['pchembl_group'].unique():
    group_data = data[data['pchembl_group'] == group]
    fingerprints = [smiles_to_fingerprint(smiles) for smiles in group_data['smiles'] if smiles]
    fingerprints_matrix = np.array([fp for fp in fingerprints if fp is not None])

    # Verificar se há amostras suficientes para t-SNE
    if len(fingerprints_matrix) > 5:  # Ajuste este valor conforme necessário
        # Aplicar t-SNE
        perplexity = min(30, len(fingerprints_matrix) - 1)
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        tsne_result = tsne.fit_transform(fingerprints_matrix)
        tsne_results.extend(tsne_result)
        group_labels.extend([group] * len(tsne_result))

# Converter resultados em DataFrame
tsne_df = pd.DataFrame(tsne_results, columns=['x', 'y'])
tsne_df['group'] = group_labels

# Plotar gráfico scatter
plt.figure(figsize=(12, 8))
for group in tsne_df['group'].unique():
    subset = tsne_df[tsne_df['group'] == group]
    plt.scatter(subset['x'], subset['y'], label=group)

plt.legend()
plt.title('Distribuição dos Ligantes por Grupo de pChEMBL Value')
plt.xlabel('t-SNE feature 0')
plt.ylabel('t-SNE feature 1')
plt.show()

