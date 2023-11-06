from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

# Função para processar um chunk de dados
def process_chunk(chunk, threshold):
    # Converter SMILES em objetos de molécula RDKit e calcular fingerprints
    chembl_molecules = (Chem.MolFromSmiles(smile) for smile in chunk['chembl_smile'])
    pkidb_molecules = (Chem.MolFromSmiles(smile) for smile in chunk['pkidb_smile'])

    chembl_fps = (AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in chembl_molecules if mol)
    pkidb_fps = (AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in pkidb_molecules if mol)

    # Calcular as similaridades de Tanimoto
    tanimoto_similarities = [DataStructs.FingerprintSimilarity(chembl_fp, pkidb_fp)
                             for chembl_fp, pkidb_fp in zip(chembl_fps, pkidb_fps)]

    # Adicionar as similaridades de Tanimoto ao chunk DataFrame
    chunk['tanimoto_similarity'] = tanimoto_similarities

    # Filtrar com base no limiar de similaridade de Tanimoto
    filtered_chunk = chunk[chunk['tanimoto_similarity'] >= threshold]

    return chunk, filtered_chunk

# Função principal
def main():
    file_path = './similar_molecules_3.tsv'
    threshold = 0.6
    chunksize = 10000  # Ajuste este valor de acordo com a sua memória disponível


    # Preparar os dataframes finais
    all_data = pd.DataFrame()
    filtered_data = pd.DataFrame()

    # Processar os dados em chunks usando paralelização
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        # Iniciar tqdm para acompanhar o número total de chunks
        total_chunks = sum(1 for _ in pd.read_csv(file_path, sep='\t', chunksize=chunksize))
        tqdm_iter = tqdm(pd.read_csv(file_path, sep='\t', chunksize=chunksize), total=total_chunks, desc="Processing Chunks")

        for chunk in tqdm_iter:
            futures.append(executor.submit(process_chunk, chunk, threshold))

        # Iniciar tqdm para acompanhar o progresso da conclusão dos futuros
        for future in tqdm(futures, total=len(futures), desc="Completing Futures"):
            chunk, filtered_chunk = future.result()
            all_data = pd.concat([all_data, chunk], ignore_index=True)
            filtered_data = pd.concat([filtered_data, filtered_chunk], ignore_index=True)

    # Debug: imprimir o número de linhas antes e após o filtro
    print(f"Número de linhas antes do filtro: {len(all_data)}")
    print(f"Número de linhas após o filtro: {len(filtered_data)}")

