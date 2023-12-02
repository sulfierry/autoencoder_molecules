import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor

def tanimoto_similarity(fp1, fp2):
    return DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)
