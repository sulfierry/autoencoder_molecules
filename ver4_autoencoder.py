import torch
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd
import faiss
from tqdm.auto import tqdm
import time

# Definindo o dispositivo como GPU (CUDA) se disponível, senão será CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path, pkidb_file_path, device=device):
        """
        Inicializa o objeto MoleculeSimilarityFinder.

        :param chembl_file_path: Caminho para o arquivo CSV que contém dados do ChEMBL.
        :param pkidb_file_path: Caminho para o arquivo CSV que contém dados do PKIDB.
        :param device: Dispositivo no qual rodar o modelo (CPU ou CUDA).
        """
        self.chembl_file_path = chembl_file_path
        self.pkidb_file_path = pkidb_file_path
        self.device = device

        # Carrega o tokenizador e o modelo ChemBERTa pré-treinado
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        
        # Move o modelo para o dispositivo especificado
        self.model.to(device)

    def get_molecule_embedding(self, smiles_list):
        """
        Gera embeddings para uma lista de moléculas representadas como SMILES.

        :param smiles_list: Lista de strings SMILES representando moléculas.
        :return: Embeddings das moléculas.
        """
        # Tokeniza os SMILES e move os tokens para o dispositivo especificado
        tokens = self.tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        
        # Gera as embeddings sem calcular gradientes (para economizar memória e acelerar)
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        # Calcula a média das embeddings em todas as posições para obter um único vetor por molécula
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def find_similar_molecules(self, threshold=0.8, batch_size=1024):
        """
        Encontra moléculas similares entre os conjuntos de dados ChEMBL e PKIDB.

        :param threshold: Limiar de similaridade para considerar duas moléculas como similares.
        :param batch_size: Número de moléculas a serem processadas por vez.
        :return: Scores de similaridade e informações sobre moléculas similares.
        """
        # Carrega os dados do ChEMBL e do PKIDB
        chembl_data = pd.read_csv(self.chembl_file_path, sep='\t')
        pkidb_data = pd.read_csv(self.pkidb_file_path, header=None, names=['pkidb_smile'])

        # Extrai SMILES e nomes (se disponíveis) das moléculas do ChEMBL
        chembl_smiles = chembl_data['canonical_smiles'].tolist()
        chembl_names = chembl_data.get('chembl_name', ['Unknown'] * len(chembl_smiles))
        
        # Extrai SMILES das moléculas do PKIDB
        pkidb_smiles = pkidb_data['pkidb_smile'].tolist()

        # Inicializa listas para armazenar resultados
        similar_molecules_info = []
        similarity_scores = []

        # Itera sobre as moléculas do PKIDB em lotes
        for i in tqdm(range(0, len(pkidb_smiles), batch_size), desc="Processing PKIDB molecules"):
            # Gera embeddings para o lote atual de moléculas do PKIDB
            pkidb_batch = pkidb_smiles[i:i + batch_size]
            pkidb_embeddings = self.get_molecule_embedding(pkidb_batch).cpu().numpy()
            faiss.normalize_L2(pkidb_embeddings)

            # Itera sobre as moléculas do ChEMBL em lotes
            for j in tqdm(range(0, len(chembl_smiles), batch_size), desc="Processing ChEMBL molecules", leave=False):
                # Gera embeddings para o lote atual de moléculas do ChEMBL
                chembl_batch = chembl_smiles[j:j + batch_size]
                chembl_embeddings = self.get_molecule_embedding(chembl_batch).cpu().numpy()
                faiss.normalize_L2(chembl_embeddings)

                # Cria um índice FAISS para buscar moléculas similares rapidamente
                index = faiss.IndexFlatIP(chembl_embeddings.shape[1])
                index.add(chembl_embeddings)

                # Busca as moléculas mais similares
                D, I = index.search(pkidb_embeddings, k=chembl_embeddings.shape[0])

                # Armazena as informações sobre moléculas similares
                for k in range(pkidb_embeddings.shape[0]):
                    for l, score in zip(I[k], D[k]):
                        if score > threshold:
                            similarity_scores.append(score)
                            similar_molecules_info.append({
                                'molregno': chembl_data.loc[j + l, 'molregno'],
                                'chembl_name': chembl_names[j + l],
                                'chembl_smile': chembl_batch[l],
                                'pkidb_smile': pkidb_batch[k],
                                'similarity': score
                            })

        return similarity_scores, similar_molecules_info

def save_similar_molecules_to_tsv(similar_molecules_info, file_path):
    """
    Salva as moléculas similares encontradas em um arquivo TSV.

    :param similar_molecules_info: Informações sobre moléculas similares.
    :param file_path: Caminho para o arquivo de saída.
    """
    df = pd.DataFrame(similar_molecules_info)
    df.to_csv(file_path, sep='\t', index=False)

def main():
    """
    Função principal para executar o programa.
    """
    # Define os caminhos para os arquivos de entrada e saída
    chembl_file_path = '/content/molecules_with_bio_activities.tsv'
    pkidb_file_path = '/content/smiles_from_pkidb_to_rdkit.tsv'
    output_file_path = '/content/similar_molecules.tsv'
    
    # Define o limiar de similaridade e o tamanho do lote
    threshold = 0.8
    batch_size = 256

    # Cria um objeto MoleculeSimilarityFinder e executa a busca por moléculas similares
    similarity_finder = MoleculeSimilarityFinder(chembl_file_path, pkidb_file_path, device)
    print("Usando o dispositivo:", device)

    similarity_scores, similar_molecules_info = similarity_finder.find_similar_molecules(threshold, batch_size)
    
    # Salva os resultados em um arquivo TSV
    save_similar_molecules_to_tsv(similar_molecules_info, output_file_path)
