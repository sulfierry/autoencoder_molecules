import torch
import faiss
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from transformers import RobertaModel, RobertaTokenizer

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


    def get_molecule_embedding(self, smiles_list, inner_batch_size=32):
        """
        Gera embeddings para uma lista de moléculas representadas como SMILES.

        :param smiles_list: Lista de strings SMILES representando moléculas.
        :return: Embeddings das moléculas.
        """
        all_embeddings = []
        for i in range(0, len(smiles_list), inner_batch_size):
            smiles_batch = smiles_list[i:i + inner_batch_size]

            # Tokeniza os SMILES e move os tokens para o dispositivo especificado
            tokens = self.tokenizer(smiles_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            tokens = {key: val.to(self.device) for key, val in tokens.items()}

            # Gera as embeddings sem calcular gradientes (para economizar memória e acelerar)
            with torch.no_grad():
                outputs = self.model(**tokens)

            # Calcula a média das embeddings em todas as posições para obter um único vetor por molécula
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu())  # Move embeddings to CPU to save GPU memory

            del tokens, outputs, embeddings
            torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)


    def find_similar_molecules(self, threshold=0.8, batch_size=64, inner_batch_size=32):
        """
        Encontra moléculas similares com base em embeddings moleculares.

        :param threshold: Limiar de similaridade para considerar as moléculas como similares
        :param batch_size: Número de moléculas a serem processadas por lote
        :param inner_batch_size: Tamanho do lote interno para processamento de embeddings
        :return: Uma lista de pontuações de similaridade e uma lista de informações sobre as moléculas similares encontradas
        """

        # Carregar os dados do ChEMBL e PKIDB
        chembl_data = pd.read_csv(self.chembl_file_path, sep='\t')
        pkidb_data = pd.read_csv(self.pkidb_file_path, header=None, names=['pkidb_smile'])

        # Converter as colunas de SMILES em listas para processamento em lote
        chembl_smiles = chembl_data['canonical_smiles'].tolist()
        pkidb_smiles = pkidb_data['pkidb_smile'].tolist()

        # Inicializar listas para armazenar informações sobre moléculas similares e pontuações de similaridade
        similar_molecules_info = []
        similarity_scores = []

        # Loop sobre os lotes de moléculas do PKIDB
        for i in tqdm(range(0, len(pkidb_smiles), batch_size), desc="Processing PKIDB molecules"):
            # Selecionar o lote atual de SMILES do PKIDB
            pkidb_batch = pkidb_smiles[i:i + batch_size]

            # Calcular embeddings para o lote atual de moléculas do PKIDB
            with torch.no_grad():  # Desativar cálculo de gradientes para economizar memória
                pkidb_embeddings = self.get_molecule_embedding(pkidb_batch, inner_batch_size=inner_batch_size).cpu().numpy()

            # Normalizar os embeddings para calcular a similaridade do cosseno
            faiss.normalize_L2(pkidb_embeddings)

            # Loop sobre os lotes de moléculas do ChEMBL
            for j in tqdm(range(0, len(chembl_smiles), batch_size), desc="Processing ChEMBL molecules", leave=False):
                # Selecionar o lote atual de SMILES do ChEMBL
                chembl_batch = chembl_smiles[j:j + batch_size]

                # Calcular embeddings para o lote atual de moléculas do ChEMBL
                with torch.no_grad():
                    chembl_embeddings = self.get_molecule_embedding(chembl_batch, inner_batch_size=inner_batch_size).cpu().numpy()

                # Normalizar os embeddings
                faiss.normalize_L2(chembl_embeddings)

                # Criar um índice FAISS para buscar moléculas similares
                index = faiss.IndexFlatIP(chembl_embeddings.shape[1])
                index.add(chembl_embeddings)

                # Procurar as moléculas mais similares do ChEMBL para cada molécula do PKIDB
                D, I = index.search(pkidb_embeddings, k=chembl_embeddings.shape[0])

                # Loop sobre os resultados da busca
                for k, (indices, distances) in enumerate(zip(I, D)):
                    for idx, score in zip(indices, distances):
                        if score > threshold:
                            # Se a similaridade for maior que o limiar, adicione as informações à lista
                            similar_molecules_info.append({
                                'molregno': chembl_data.loc[j + idx, 'molregno'],
                                'chembl_smile': chembl_batch[idx],
                                'pkidb_smile': pkidb_batch[k],
                                'similarity': score
                            })
                            similarity_scores.append(score)

                # Liberar a memória dos tensores e da GPU
                del chembl_embeddings
                torch.cuda.empty_cache()

            # Liberar a memória dos tensores e da GPU
            del pkidb_embeddings
            torch.cuda.empty_cache()

        return similarity_scores, similar_molecules_info

class MoleculeVisualization:
    def __init__(self):
        pass

    @staticmethod
    def plot_histogram(similarity_scores):
        """
        Plota um histograma dos scores de similaridade.
        :param similarity_scores: Lista de scores de similaridade entre as moléculas.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(similarity_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("Distribuição dos Scores de Similaridade")
        plt.xlabel("Score de Similaridade")
        plt.ylabel("Número de Moléculas")
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    @staticmethod
    def visualize_with_tsne(embeddings, labels):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis", s=100, alpha=0.7)
        plt.title('t-SNE Visualization of Molecule Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

    @staticmethod
    def cluster_and_visualize(embeddings, num_clusters=5):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
        labels = kmeans.labels_

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis", s=100, alpha=0.7)
        plt.title('Clustered Visualization of Molecule Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

    @staticmethod
    def save_similar_molecules_to_tsv(similar_molecules_info, file_path):
        """
        Salva as moléculas similares encontradas em um arquivo TSV.
        :param similar_molecules_info: Informações sobre moléculas similares.
        :param file_path: Caminho para o arquivo de saída.
        """
        df = pd.DataFrame(similar_molecules_info)
        df.to_csv(file_path, sep='\t', index=False)


def main():
    chembl_file_path = '/content/molecules_with_bio_activities.tsv'
    pkidb_file_path = '/content/smiles_from_pkidb_to_rdkit.tsv'
    output_file_path = '/content/similar_molecules_3.tsv'
    threshold = 0.8
    
    print("Usando o dispositivo:", device)

    # Inicializa o objeto MoleculeSimilarityFinder
    similarity_finder = MoleculeSimilarityFinder(chembl_file_path, pkidb_file_path, device)

    # Encontra moléculas similares e retorna pontuações e informações
    similarity_scores, similar_molecules_info = similarity_finder.find_similar_molecules(threshold)

    # Instanciando o visualizador de moléculas
    molecule_visualizer = MoleculeVisualization()

    # Salva as moléculas similares encontradas em um arquivo TSV
    molecule_visualizer.save_similar_molecules_to_tsv(similar_molecules_info, output_file_path)

    # Carrega os embeddings para visualização
    chembl_data = pd.read_csv(chembl_file_path, sep='\t')
    pkidb_data = pd.read_csv(pkidb_file_path, header=None, names=['pkidb_smile'])
    chembl_smiles = chembl_data['canonical_smiles'].tolist()
    pkidb_smiles = pkidb_data['pkidb_smile'].tolist()

    chembl_embeddings = similarity_finder.get_molecule_embedding(chembl_smiles).cpu().numpy()
    pkidb_embeddings = similarity_finder.get_molecule_embedding(pkidb_smiles).cpu().numpy()

    all_embeddings = np.concatenate([chembl_embeddings, pkidb_embeddings], axis=0)

    # Visualização com t-SNE
    molecule_visualizer.visualize_with_tsne(all_embeddings, ['ChEMBL']*len(chembl_embeddings) + ['PKIDB']*len(pkidb_embeddings))

    # Clusterização e Visualização
    molecule_visualizer.cluster_and_visualize(all_embeddings, num_clusters=7)

    # Histograma de Scores de Similaridade
    molecule_visualizer.plot_histogram(similarity_scores)

if __name__ == "__main__":
    main()
