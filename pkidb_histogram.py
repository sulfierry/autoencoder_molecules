import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os conjuntos de dados
pkidb_data = pd.read_csv('../PKIDB/pkidb_2023-06-30.tsv', sep='\t')
out_pkidb_data = pd.read_csv('out_pkidb.tsv', sep='\t')
