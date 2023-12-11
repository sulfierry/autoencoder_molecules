from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# Carregar seu arquivo TSV
data = pd.read_csv('../1_database/nr_kinase_drug_info_kd_ki_ic50_10uM_manually_validated.tsv', sep='\t')

# Função para converter SMILES em arquivos .sdf
def smiles_to_sdf(smiles, file_name):
    # Converter SMILES para um objeto de molécula
    mol = Chem.MolFromSmiles(smiles)

    # Se a molécula for válida, salve-a como um arquivo .sdf
    if mol:
        # Adicionando hidrogênios (opcional)
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        # Criar um escritor SDF e escrever a molécula
        with Chem.SDWriter(file_name) as writer:
            writer.write(mol)

# Converter cada SMILES da coluna 'canonical_smiles' em um arquivo .sdf
for index, row in data.iterrows():
    smiles = row['canonical_smiles']
    molregno = row['molregno']
    file_name = f'mol_{molregno}.sdf'
    smiles_to_sdf(smiles, file_name)
