import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import concurrent.futures
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import cosine
import concurrent.futures
import os

class Histogram:
    def __init__(self, valid_fps, batch_size):
        self.valid_fps = valid_fps
        self.batch_size = batch_size

    @staticmethod
    def process_batch(fingerprints, calculate_function):
        """
        Processa um lote de fingerprints utilizando a função de cálculo fornecida.
        """
        results = []
        for fp in fingerprints:
            results.extend(calculate_function(fp, fingerprints))
        return results

    @staticmethod
    def process_batch_helper(batch, process_function, fingerprints, metric):
        """
        Função auxiliar para processar um lote de fingerprints. Esta função é utilizada
        para evitar problemas de serialização em ambientes de processamento paralelo.
        """
        return Histogram.process_batch(batch, lambda fp: process_function(fp, fingerprints, metric))
       
        
    def process_in_batches_parallel(self, process_function, metric):
        """
        Processa os dados em lotes, de forma paralela, utilizando todos os CPUs disponíveis.
        """
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for i in range(0, len(self.valid_fps), self.batch_size):
                batch = self.valid_fps[i:i+self.batch_size]
                futures.append(executor.submit(process_batch_parallel, batch, process_function, self.valid_fps, metric))
            
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results


    def plot_histograms(self, all_similarities, all_distances, data, plt):
        """
        Plota os histogramas das similaridades e distâncias calculadas, além do histograma de valores pChEMBL.
        """
        fig, axs = plt.subplots(3, 2, figsize=(13, 13))  # 3 linhas, 2 colunas

        # Plotar histogramas das similaridades
        for ax, (metric, values) in zip(axs[:, 0], all_similarities.items()):
            ax.hist(values, bins=50, color='skyblue', edgecolor='black')
            ax.set_title(f'Similarity - {metric.capitalize()}', fontsize=10)
            ax.set_ylabel('Frequency')

        # Plotar histogramas das distâncias
        for ax, (metric, values) in zip(axs[:, 1], all_distances.items()):
            ax.hist(values, bins=50, color='salmon', edgecolor='black')
            ax.set_title(f'Distance - {metric.capitalize()}', fontsize=10)
            ax.set_ylabel('Frequency')

        # Plotar histograma dos valores de pChEMBL no último subplot
        axs[2, 1].hist(data['pchembl_value'].dropna(), bins=20, color='green', edgecolor='black')
        axs[2, 1].set_title('ChEMBL value', fontsize=10)
        axs[2, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('histogram_similarity_distance.png')
        plt.show()
