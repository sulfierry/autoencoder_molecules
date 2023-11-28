import csv
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from tqdm import tqdm

def process_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            MW = Descriptors.MolWt(mol)
            CLogP = Crippen.MolLogP(mol)
            HBD = Lipinski.NumHDonors(mol)
            HBA = Lipinski.NumHAcceptors(mol)
            TPSA = rdMolDescriptors.CalcTPSA(mol)
            NRB = Lipinski.NumRotatableBonds(mol)
            NAR = rdMolDescriptors.CalcNumAromaticRings(mol)
            NCA = rdMolDescriptors.CalcNumRings(mol)

            # Log das propriedades computadas
            print(f"SMILES: {smiles}, MW: {MW}, CLogP: {CLogP}, HBD: {HBD}, HBA: {HBA}, TPSA: {TPSA}, NRB: {NRB}, NAR: {NAR}, NCA: {NCA}")

            # Verificar as condições de filtragem
            if (314 <= MW <= 613) and (0.7 <= CLogP <= 6.3) and (0 <= HBD <= 4) and (3 <= HBA <= 10) and \
               (55 <= TPSA <= 138) and (1 <= NRB <= 11) and (1 <= NAR <= 5) and (0 <= NCA <= 2):
                return (True, smiles, MW, CLogP, HBD, HBA, TPSA, NRB, NAR, NCA)
        return (False, None, None, None, None, None, None, None, None, None)
    except Exception as e:
        print(f"Error processing SMILES: {smiles}: {e}")
        return (False, None, None, None, None, None, None, None, None, None)


if __name__ == "__main__":
    input_file = "./chembl_Kd_Ki/tanimoto_filtered_similar_molecules.tsv"
    output_file = "pkidb_pkiFilter_detailed_out2.tsv"

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        reader = csv.DictReader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        
        # Escrever cabeçalho
        writer.writerow([
            "chembl_smiles", "chembl_MW", "chembl_CLogP", "chembl_HBD", "chembl_HBA", "chembl_TPSA", "chembl_NRB", "chembl_NAR", "chembl_NCA",
            "target_smiles", "target_MW", "target_CLogP", "target_HBD", "target_HBA", "target_TPSA", "target_NRB", "target_NAR", "target_NCA"
        ])
        
        for row in tqdm(reader):
            # Processar chembl_smiles
            chembl_result = process_molecule(row['chembl_smile'])
            # Processar target_smiles
            target_result = process_molecule(row['target_smile'])
            
            # Se ambas as moléculas são válidas, escrever lado a lado
            if chembl_result[0] and target_result[0]:
                writer.writerow(chembl_result[1:] + target_result[1:])
