# autoecoder_exploring

# Molecule Similarity Finder

Este código é destinado a encontrar moléculas similares entre dois conjuntos de dados utilizando um autoencoder para extrair características latentes das strings SMILES das moléculas.

## Como funciona

1. **Dados de Entrada**: O programa aceita dois conjuntos de dados - um do ChEMBL e outro do PKIDB. Ambos devem conter strings SMILES, que são representações textuais padrão de estruturas de moléculas.
2. **Preprocessamento**: As strings SMILES são preprocessadas para garantir a consistência e remover valores nulos. Além disso, as moléculas com strings SMILES muito longas são filtradas para manter a dimensão dos dados gerenciável.
3. **Embeddings**: As strings SMILES são convertidas em embeddings utilizando uma camada de embedding do PyTorch.
4. **Autoencoder**: Um autoencoder é definido e treinado nos embeddings das strings SMILES para extrair características latentes das moléculas. O autoencoder consiste de uma camada de encoder LSTM seguida por uma camada de decoder LSTM. O objetivo é reconstruir a entrada a partir da representação latente.
5. **Cálculo de Similaridade**: Depois que o autoencoder é treinado, usamos a camada de encoder para obter as características latentes para todas as moléculas nos dois conjuntos de dados. A semelhança entre as moléculas é então calculada usando similaridade de cosseno entre essas características latentes.
6. **Resultados**: O programa retorna os índices das moléculas mais similares do conjunto de dados PKIDB para cada molécula no conjunto de dados ChEMBL.

## Dependências

- **os**
- **sys**
- **numpy**
- **pandas**
- **torch**
- **sklearn**

