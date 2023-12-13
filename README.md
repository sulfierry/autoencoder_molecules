# SMILES Autoencoder (ver_0.7_pkidb_chembl.py)

## Description

The Molecular Similarity Finder is a Python script designed for molecular similarity analysis using the ChemBERTa model. This script enables users to identify molecules that are similar to a given target dataset based on their Simplified Molecular Input Line Entry System (SMILES) representations. It leverages parallel processing for efficient molecular embedding calculations and employs the FAISS library for fast and scalable similarity searches.

## How it Works

1. **Input Data**: The program requires two datasets: one from ChEMBL and one from PKIDB. Both datasets must contain SMILES notations, which are standardized textual representations of molecular structures.

2. **Preprocessing**: SMILES strings are preprocessed to ensure consistency and to remove any null or invalid values. Preprocessing also includes tokenizing the SMILES strings, preparing them to be processed by the ChemBERTa model.

3. **Generating Embeddings**: Using the pre-trained ChemBERTa model, SMILES strings are transformed into embeddings, which are dense vector representations capturing the essential features of the molecules.

4. **Normalization**: The embeddings are normalized to ensure that the calculated similarity measure is robust and accurately reflects the structural similarities between the molecules.

5. **Similarity Calculation**: Similarity between molecules from the two datasets is calculated using cosine similarity between their normalized embeddings. For each molecule in the PKIDB dataset, we identify the most similar molecules in the ChEMBL dataset.

6. **Results**: The program returns a list of the most similar molecules, including their SMILES notations, indices, and similarity scores. The results can be used for subsequent analyses, such as studies of structure-activity relationships or to identify potential drug candidates.

## Key Features

- **Data Preprocessing:** Load and preprocess molecular data from ChemBL (Chemical Biology Database) and a user-defined target dataset.
- **Embedding Generation:** Calculate molecular embeddings for both datasets using the state-of-the-art ChemBERTa model.
- **Similarity Search:** Perform rapid similarity search across molecules using FAISS, enabling the identification of structurally similar compounds.
- **Visualization:** Visualize the results through dimensionality reduction using t-SNE (t-Distributed Stochastic Neighbor Embedding) and cluster analysis.
- **Score Histograms:** Generate histograms to visualize the distribution of similarity scores among the identified molecules.
- **Data Export:** Save the details of similar molecules to a structured TSV (Tab-Separated Values) file for further analysis.

## Usage Instructions

To utilize the Molecular Similarity Finder:

1. **Install Dependencies:** Ensure that the necessary Python libraries, including PyTorch, FAISS, Transformers, NumPy, Pandas, Seaborn, Matplotlib, and Scikit-learn, are installed in your environment.

2. **Customize Configuration:** Tailor the script to your specific dataset and preferences by adjusting file paths, similarity thresholds, and other configurable parameters.

3. **Execution:** Execute the script. It will handle the computation of molecular embeddings, conduct similarity searches, and visualize the results.

4. **Results:** Examine the output files, including similarity scores and visualizations, generated as per your script's settings.

## Dependencies

The script relies on a selection of widely-used Python libraries:

- **PyTorch:** For deep learning and neural network operations.
- **FAISS:** To enable efficient similarity search.
- **Transformers (Hugging Face):** For working with the ChemBERTa model.
- **NumPy:** For numerical and array operations.
- **Pandas:** For data manipulation and storage.
- **Seaborn:** For enhanced data visualization.
- **Matplotlib:** For creating plots and figures.
- **Scikit-learn:** For machine learning and clustering tasks.

Ensure that these libraries are installed in your Python environment before executing the script.

## CVAE for SMILES Data Processing (cvae_0.03.py)

This code implements a Conditional Variational Autoencoder (CVAE) for processing and generating chemical structures represented in SMILES notation. The primary objective is to learn a generative model of chemical compounds using a deep learning approach.

## Modules and Libraries
- `torch`: PyTorch library for tensor computations and neural network operations.
- `pandas`: Data manipulation and analysis library.
- `rdkit`: Collection of cheminformatics and machine learning tools.
- `matplotlib.pyplot`: Library for creating static, animated, and interactive visualizations.
- `transformers`: Hugging Face's library for state-of-the-art Natural Language Processing.
- `sklearn.model_selection`: Scikit-learn module for splitting data arrays into train and test subsets.
- `concurrent.futures`: Provides a high-level interface for asynchronously executing callables.

## Configuration
- `DEVICE`: Specifies the device for computation (CPU or GPU).
- `NUM_CPUS`: Number of CPU cores for parallel processing.
- `EPOCHS`, `BATCH_SIZE`, `LATENT_DIM`, `LEARNING_RATE`, `LOG_INTERVAL`, `WEIGHT_DECAY`: Hyperparameters for the model and training process.

## CVAE Model
A PyTorch-based implementation of a Conditional Variational Autoencoder:
- **Encoder**: Uses a pre-trained RobertaModel to encode the input SMILES data.
- **Latent Space**: Maps the encoded representation to a latent space using linear transformations (`fc_mu` and `fc_var`).
- **Decoder**: Reconstructs SMILES from the latent space representation.

## Training Function (`train_cvae`)
Handles the training process of the CVAE model. It freezes the encoder parameters for fine-tuning, uses gradient scaling for mixed precision training, and computes the training loss using backpropagation.

## DataLoader and Dataset
- **SmilesDataset**: Custom dataset class for storing and tokenizing SMILES data. It uses parallel tokenization for efficiency.
- **data_pre_processing**: Function to prepare DataLoader objects for training, validation, and testing phases.

## Utility Functions
- `smiles_to_token_ids_parallel`: Converts SMILES strings to token IDs in parallel.
- `token_ids_to_smiles`: Decodes token IDs back to SMILES strings.
- `postprocess_smiles`: Post-processes generated SMILES for validation and analysis.
- `calculate_properties`, `is_similar`: Functions to calculate chemical properties and assess similarity.

## Main Function
Executes the entire pipeline:
1. Loads and preprocesses SMILES data.
2. Initializes and trains the CVAE model.
3. Saves the trained model and outputs training progress.

## Usage
To run the code, ensure all dependencies are installed and execute the script. The model will train on the specified SMILES dataset and save the trained model for future use.

---

This implementation serves as a robust framework for exploring generative models in cheminformatics, enabling the generation of novel chemical structures with desired properties.


## Contributing

Contributions to improve the model or extend its functionalities are welcome. Please follow the standard pull request process for contributions.


## License and Usage

The Molecular Similarity Finder script is open-source and provided under a permissive license, granting users the freedom to use and modify it as needed. For comprehensive details regarding its usage and licensing terms, please refer to the script's header section.

---

**Important Note:** Prior to executing the script, it is advisable to thoroughly review its content and tailor it to suit your specific research or application. Detailed instructions and configuration options are comprehensively documented within the script itself.
