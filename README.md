# SMILES Autoencoder

## Descrição

O objetivo deste projeto é explorar diferentes arquiteturas de autoencoders para aprender representações latentes de moléculas. 
Estas representações podem ser utilizadas para diversas aplicações, como descoberta de fármacos, otimização de moléculas e muito mais.

## Molecule Similarity Finder

O código é projetado para identificar moléculas similares entre dois conjuntos de dados distintos, utilizando um modelo pré-treinado de ChemBERTa para extrair representações vetoriais (embeddings) dos SMILES. Estas representações são então utilizadas para calcular a similaridade entre as moléculas dos dois conjuntos.

## Como funciona

1. **Dados de Entrada**: O programa requer dois conjuntos de dados: um do ChEMBL e outro do PKIDB. Ambos os conjuntos de dados devem conter notações SMILES, que são representações textuais padronizadas das estruturas moleculares.

2. **Preprocessamento**: As strings SMILES são preprocessadas para assegurar consistência e para remover quaisquer valores nulos ou inválidos. O preprocessamento também inclui a tokenização das strings SMILES, preparando-as para serem processadas pelo modelo ChemBERTa.

3. **Geração de Embeddings**: Utilizando o modelo ChemBERTa pré-treinado, as strings SMILES são transformadas em embeddings, que são representações vetoriais densas capturando as características essenciais das moléculas.

4. **Normalização**: Os embeddings são normalizados para garantir que a medida de similaridade calculada seja robusta e reflita adequadamente as semelhanças estruturais entre as moléculas.

5. **Cálculo de Similaridade**: A similaridade entre as moléculas dos dois conjuntos de dados é calculada usando a similaridade de cosseno entre seus embeddings normalizados. Para cada molécula no conjunto de dados PKIDB, identificamos as moléculas mais similares no conjunto de dados ChEMBL.

6. **Resultados**: O programa retorna uma lista das moléculas mais similares, incluindo suas notações SMILES, índices e pontuações de similaridade. Os resultados podem ser utilizados para análises subsequentes, como estudos de relações estrutura-atividade ou para identificar potenciais candidatos a fármacos.


## Dependências

- **os**
- **sys**
- **numpy**
- **pandas**
- **torch**
- **sklearn**

