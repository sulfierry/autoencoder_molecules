import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import cosine
import seaborn as sns

"""
	O presente algoritmo compara todos os fingerprints moleculares uns contra os outros, 
	o que é conhecido como uma abordagem "todos contra todos". Para cada fingerprint no 
	conjunto de dados,ele calcula a similaridade ou a distância com todos os outros 
	fingerprints. Este processo é repetido para cada fingerprint, resultando em um 
	conjunto abrangente de medidas de similaridade e distância entre cada par de moléculas 
	representadas pelos fingerprints.

	Essa abordagem pode ser computacionalmente intensiva, especialmente para conjuntos de dados 
	grandes, porque o número de comparações cresce quadraticamente com o número de fingerprints. 
	Por exemplo, se há N fingerprints, o número de comparações será N×(N−1)/2, o que pode ser um 
	número muito grande para grandes conjuntos de dados. É por isso que o algoritmo processa os 
	dados em lotes, para gerenciar o uso de memória e recursos de processamento.
"""


class Histogram:
    def __init__(self, file_path, batch_size=1024):
        self.file_path = file_path
        self.batch_size = batch_size
