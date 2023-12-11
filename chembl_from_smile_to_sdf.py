from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

# Carregar seu arquivo TSV
data = pd.read_csv('../1_database/nr_kinase_drug_info_kd_ki_ic50_10uM_manually_validated.tsv', sep='\t')


# Criar uma pasta para os arquivos .sdf se não existir
output_folder = 'smiles_to_sdf'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Função para converter SMILES em arquivos .sdf
def smiles_to_sdf(smiles, molregno):
    file_name = os.path.join(output_folder, f'mol_{molregno}.sdf')
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
