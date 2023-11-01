# SMILES Autoencoder

## Description

## Molecule Similarity Finder

The code is designed to identify similar molecules between two distinct datasets, using a pre-trained ChemBERTa model to extract vector representations (embeddings) from SMILES notations. These representations are then used to calculate the similarity between the molecules of the two sets.

## How it Works

1. **Input Data**: The program requires two datasets: one from ChEMBL and one from PKIDB. Both datasets must contain SMILES notations, which are standardized textual representations of molecular structures.

2. **Preprocessing**: SMILES strings are preprocessed to ensure consistency and to remove any null or invalid values. Preprocessing also includes tokenizing the SMILES strings, preparing them to be processed by the ChemBERTa model.

3. **Generating Embeddings**: Using the pre-trained ChemBERTa model, SMILES strings are transformed into embeddings, which are dense vector representations capturing the essential features of the molecules.

4. **Normalization**: The embeddings are normalized to ensure that the calculated similarity measure is robust and accurately reflects the structural similarities between the molecules.

5. **Similarity Calculation**: Similarity between molecules from the two datasets is calculated using cosine similarity between their normalized embeddings. For each molecule in the PKIDB dataset, we identify the most similar molecules in the ChEMBL dataset.

6. **Results**: The program returns a list of the most similar molecules, including their SMILES notations, indices, and similarity scores. The results can be used for subsequent analyses, such as studies of structure-activity relationships or to identify potential drug candidates.

## Dependencies

- **os**
- **sys**
- **numpy**
- **pandas**
- **torch**
- **sklearn**
