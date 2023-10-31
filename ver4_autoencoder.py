import torch
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd
import numpy as np
import faiss
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Escolha o dispositivo disponível (CUDA ou CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class MoleculeSimilarityFinder:
    def __init__(self, chembl_file_path, pkidb_file_path, device=device):
        self.chembl_file_path = chembl_file_path
        self.pkidb_file_path = pkidb_file_path
        self.device = device

        # Carregar o modelo ChemBERTa e o tokenizador
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')

        # Mova o modelo para o dispositivo correto
        self.model.to(device)

    def load_data(self, file_path):
        data = pd.read_csv(file_path, sep='\t', header=None)
        smiles = data[0].tolist()
        return smiles

    def get_molecule_embedding(self, smiles_list):
        tokens = self.tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def find_similar_molecules(self, threshold=0.8, batch_size=1024):
        chembl_data = pd.read_csv(self.chembl_file_path, sep='\t')
        pkidb_data = pd.read_csv(self.pkidb_file_path, header=None, names=['pkidb_smile'])

        chembl_smiles = chembl_data['canonical_smiles'].tolist()
        pkidb_smiles = pkidb_data['pkidb_smile'].tolist()

        similar_molecules_info = []
        similarity_scores = []  # adicionado

        for i in tqdm(range(0, len(pkidb_smiles), batch_size), desc="Processing PKIDB molecules"):
            pkidb_batch = pkidb_smiles[i:i + batch_size]
            pkidb_embeddings = self.get_molecule_embedding(pkidb_batch).cpu().numpy()
            faiss.normalize_L2(pkidb_embeddings)

            for j in range(0, len(chembl_smiles), batch_size):
                chembl_batch = chembl_smiles[j:j + batch_size]
                chembl_embeddings = self.get_molecule_embedding(chembl_batch).cpu().numpy()
                faiss.normalize_L2(chembl_embeddings)

                index = faiss.IndexFlatIP(chembl_embeddings.shape[1])
                index.add(chembl_embeddings)

                D, I = index.search(pkidb_embeddings, k=chembl_embeddings.shape[0])

                for k in range(pkidb_embeddings.shape[0]):
                    for l, score in zip(I[k], D[k]):
                        if score > threshold:
                            similarity_scores.append(score)  # coleta a pontuação de similaridade
                            similar_molecules_info.append({
                                'molregno': chembl_data.loc[j + l, 'molregno'],
                                'chembl_smile': chembl_batch[l],
                                'pkidb_smile': pkidb_batch[k],
                                'similarity': score
                            })

        return similarity_scores, similar_molecules_info  # retorno corrigido

    def visualize_with_tsne(self, similarity_matrix, similarity_scores):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(similarity_matrix)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=similarity_scores, cmap="YlGnBu")
        plt.colorbar()
        plt.show()

def plot_histogram(similarity_scores):
    plt.hist(similarity_scores, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Number of Molecules")
    plt.show()


def save_similar_molecules_to_tsv(similar_molecules_info, file_path):
    df = pd.DataFrame(similar_molecules_info)
    df.to_csv(file_path, sep='\t', index=False)

def plot_similarity_matrix(similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix.todense(), cmap="YlGnBu", vmin=0, vmax=1)
    plt.show()


from google.colab import files

def main():
    # Configuração inicial
    chembl_file_path = '/content/molecules_with_bio_activities.tsv'
    pkidb_file_path = '/content/pkidb_2023-06-30.tsv'
    output_file_path = '/content/similar_molecules.tsv'
    threshold = 0.8
    batch_size = 256  # ou outro valor que você preferir

    # Instancie a classe e inicialize com os arquivos de entrada e o tamanho do batch
    similarity_finder = MoleculeSimilarityFinder(chembl_file_path, pkidb_file_path)

    print(device)

    # Encontre moléculas similares
    similarity_scores, similar_molecules_info = similarity_finder.find_similar_molecules(threshold, batch_size)  # Adicionado batch_size

    # Salve as moléculas similares em um arquivo TSV
    save_similar_molecules_to_tsv(similar_molecules_info, output_file_path)

    # Visualizar as pontuações de similaridade em um histograma
    plot_histogram(similarity_scores)

if __name__ == "__main__":
    main()
