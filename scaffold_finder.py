import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Função para encontrar o scaffold de um SMILES
def get_scaffold(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold) if scaffold else None

# Função para calcular a similaridade de Tanimoto
def calculate_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Carregar o arquivo
file_path = 'CAMINHO_PARA_SEU_ARQUIVO.tsv'
data = pd.read_csv(file_path, sep='\t')

# Smile alvo
target_smile = "NC(=O)CC(C(=O)O)N"

# Adicionando as novas colunas
data['scaffold_smile'] = data['canonical_smiles'].apply(get_scaffold)
data['similarity_smiles'] = data['canonical_smiles'].apply(lambda x: calculate_similarity(x, target_smile))

# Salvando o resultado em um novo arquivo .tsv
output_file_path = 'output_scaffolds_similarity.tsv'
data.to_csv(output_file_path, sep='\t', index=False)
