import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform
from multiprocessing import Pool, cpu_count

class MoleculeClusterer:
    def __init__(self, smiles_file_path):
        self.smiles_file_path = smiles_file_path
        self.data = None
        self.fingerprints = []

    def load_data(self):
        self.data = pd.read_csv(self.smiles_file_path, sep='\t', usecols=['canonical_smiles'])

    @staticmethod
    def smiles_to_fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2) if mol else None

    @staticmethod
    def compute_fingerprint(smiles):
        return MoleculeClusterer.smiles_to_fingerprint(smiles)

    def parallel_generate_fingerprints(self):
        with Pool(cpu_count()) as pool:
            self.fingerprints = pool.map(self.compute_fingerprint, self.data['canonical_smiles'])
        self.fingerprints = [fp for fp in self.fingerprints if fp is not None]

    def calculate_similarity_matrix(self):
        num_fps = len(self.fingerprints)
        similarity_matrix = np.zeros((num_fps, num_fps))
        for i in range(num_fps):
            for j in range(num_fps):
                similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(self.fingerprints[i], self.fingerprints[j])
        return similarity_matrix

    def calculate_cosine_distance_matrix(self):
        fp_array = np.array([list(fp) for fp in self.fingerprints])
        cosine_sim_matrix = cosine_similarity(fp_array)
        cosine_dist_matrix = 1 - cosine_sim_matrix
        return cosine_dist_matrix

    def combine_similarity_matrices(self, tanimoto_matrix, cosine_matrix, alpha=0.5):
        combined_matrix = alpha * tanimoto_matrix + (1 - alpha) * cosine_matrix
        return combined_matrix

    def cluster_molecules(self, combined_similarity_matrix, threshold=0.8):
        clustering_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=1 - threshold)
        labels = clustering_model.fit_predict(1 - combined_similarity_matrix)
        self.data['Cluster'] = labels

    def save_clustered_data(self, output_file_path):
        self.data.to_csv(output_file_path, sep='\t', index=False)


def main():
    smiles_file_path = './nr_kinase_drug_info_kd_ki_manually_validated.tsv.tsv'
    output_file_path = './clustered_smiles.tsv'

    clusterer = MoleculeClusterer(smiles_file_path)
    clusterer.load_data()
    clusterer.parallel_generate_fingerprints()

    if clusterer.fingerprints:
        tanimoto_matrix = clusterer.calculate_similarity_matrix()
        cosine_matrix = clusterer.calculate_cosine_distance_matrix()
        combined_matrix = clusterer.combine_similarity_matrices(tanimoto_matrix, cosine_matrix, alpha=0.5)
        clusterer.cluster_molecules(combined_matrix, threshold=0.8)
        clusterer.save_clustered_data(output_file_path)
    else:
        print("Nenhum fingerprint válido foi encontrado.")

if __name__ == "__main__":
    main()
