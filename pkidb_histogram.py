import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os conjuntos de dados
pkidb_data = pd.read_csv('../PKIDB/pkidb_2023-06-30.tsv', sep='\t')
out_pkidb_data = pd.read_csv('out_pkidb.tsv', sep='\t')


# Renomear as colunas do conjunto de dados out_pkidb para corresponder às colunas do conjunto de dados pkidb
out_pkidb_data = out_pkidb_data.rename(columns={"CLogP": "LogP"})

# Configurar o estilo dos gráficos
sns.set(style="whitegrid")

# Lista de descritores para análise
descritores = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'NRB']

# Criar uma figura e um conjunto de subplots
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))

# Ajustar o espaço entre os plots
plt.tight_layout(h_pad=3, w_pad=2, rect=[0.05, 0, 1, 1])

# Adicionar a legenda do eixo Y no meio do gráfico
fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=12)


# Plotar os histogramas para cada descritor
for i, descritor in enumerate(descritores):
    # Calcular a posição do subplot
    row = i // 2
    col = i % 2
    
    # Plotar o histograma para o conjunto de dados 'pkidb'
    sns.histplot(pkidb_data[descritor], bins=30, ax=axs[row, col], kde=False, color="blue", label="pkidb" if i==0 else "")
    
    # Plotar o histograma para o conjunto de dados 'out_pkidb'
    sns.histplot(out_pkidb_data[descritor], bins=30, ax=axs[row, col], kde=False, color="orange", label="out_pkidb" if i==0 else "")
    
    # Configurar o título e os rótulos
    axs[row, col].set_title(f'Distribution of {descritor}')
    axs[row, col].set_xlabel(descritor)
    axs[row, col].set(ylabel=None)
