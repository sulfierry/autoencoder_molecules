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
    elif 1 < value < 7:
        return 'grupo1 (1 - 7)'
    elif 7 <= value < 8:
        return 'grupo2 (7 - 8)'
    elif 8 <= value < 9:
        return 'grupo3 (8 - 9)'
    elif 9 <= value < 10:
        return 'grupo4 (9 - 10)'
    elif 10 <= value < 11:
        return 'grupo5 (10 - 11)'
    else:
        return '>12'

# Funções de plotagem atualizadas com a ordem explícita dos grupos
def plot_2d_tsne(tsne_df, group_order):
    plt.figure(figsize=(12, 8))
    for group in group_order:
        if group in tsne_df['group'].unique():
            subset = tsne_df[tsne_df['group'] == group]
            plt.scatter(subset['x'], subset['y'], label=group)
    plt.legend()
    plt.title('Distribuição dos Ligantes por Grupo de pChEMBL Value (2D)')
    plt.xlabel('t-SNE feature 0')
    plt.ylabel('t-SNE feature 1')
    plt.show()

def plot_3d_tsne(tsne_df, group_order):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for group in group_order:
        if group in tsne_df['group'].unique():
            subset = tsne_df[tsne_df['group'] == group]
            ax.scatter(subset['x'], subset['y'], subset['z'], label=group)
    ax.legend()
    ax.set_title('Distribuição dos Ligantes por Grupo de pChEMBL Value (3D)')
    ax.set_xlabel('t-SNE feature 0')
    ax.set_ylabel('t-SNE feature 1')
    ax.set_zlabel('t-SNE feature 2')
    plt.show()


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
    group_data.to_csv(f'./kinases_{group}.tsv', sep='\t', index=False)  # Salvar dados do grupo

    fingerprints = [smiles_to_fingerprint(smiles) for smiles in group_data['smiles'] if smiles]
    fingerprints_matrix = np.array([fp for fp in fingerprints if fp is not None])

    # Verificar se há amostras suficientes para t-SNE
    if len(fingerprints_matrix) > 5:
        tsne = TSNE(n_components=3, random_state=0, perplexity=min(30, len(fingerprints_matrix) - 1))
        tsne_result = tsne.fit_transform(fingerprints_matrix)
        tsne_results.extend(tsne_result)
        group_labels.extend([group] * len(tsne_result))

# Converter resultados em DataFrame
tsne_df = pd.DataFrame(tsne_results, columns=['x', 'y', 'z'])
tsne_df['group'] = group_labels

# Ordem desejada para os grupos
group_order = ['grupo1 (1 - 7)', 'grupo2 (7 - 8)', 'grupo3 (8 - 9)', 'grupo4 (9 - 10)', 'grupo5 (10 - 11)', '>12']

# Chamar funções de plotagem conforme necessário
plot_2d_tsne(tsne_df, group_order)
# plot_3d_tsne(tsne_df, group_order)  # Descomente esta linha para plotar em 3D
