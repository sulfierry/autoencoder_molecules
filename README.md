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

- 
# CVAE for SMILES Molecular Structures Generation (cvae_0.02.py)

This project implements a Conditional Variational Autoencoder (CVAE) for generating molecular structures using the SMILES (Simplified Molecular Input Line Entry System) format. The model utilizes RoBERTa, a pre-trained language model, as the base for the encoder, and a custom decoder.


## Key Components

- `SmilesDataset`: Class for reading and processing SMILES data.
- `CVAE`: The core model with methods for encoding, reparameterization, and decoding.
- `train_cvae`: Function to train the CVAE.
- `generate_molecule`: Function to generate new molecular structures.


## Model Workflow

The model encodes SMILES inputs using RoBERTa, creates a latent space, and decodes to generate new SMILES structures.

## Model Parameters

- `LATENT_DIM`: Dimension of the latent space.
- `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`: Training parameters.
- `EPOCHS`: Number of training epochs.


