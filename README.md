# Conda Environment Dependencies

Below are the dependencies for the Conda environment named "py39":

```bash
echo "# This file may be used to create an environment using:" > dependencies.txt
echo "# \$ conda create --name <env> --file <this file>" >> dependencies.txt
echo "platform: linux-64" >> dependencies.txt
echo "_libgcc_mutex=0.1=main" >> dependencies.txt
echo "ca-certificates=2023.08.22=h06a4308_0" >> dependencies.txt
echo "ld_impl_linux-64=2.38=h1181459_1" >> dependencies.txt
echo "libffi=3.3=he6710b0_2" >> dependencies.txt
echo "libgcc-ng=9.1.0=hdf63c60_0" >> dependencies.txt
echo "libstdcxx-ng=9.1.0=hdf63c60_0" >> dependencies.txt
echo "ncurses=6.3=h7f8727e_2" >> dependencies.txt
echo "openssl=1.1.1w=h7f8727e_0" >> dependencies.txt
echo "pip=23.3=py39h06a4308_0" >> dependencies.txt
echo "python=3.9.12=h12debd9_1" >> dependencies.txt
echo "readline=8.1.2=h7f8727e_1" >> dependencies.txt
echo "setuptools=68.0.0=py39h06a4308_0" >> dependencies.txt
echo "sqlite=3.38.5=hc218d9a_0" >> dependencies.txt
echo "tk=8.6.12=h1ccaba5_0" >> dependencies.txt
echo "tzdata=2023c=h04d1e81_0" >> dependencies.txt
echo "wheel=0.41.2=py39h06a4308_0" >> dependencies.txt
echo "xz=5.2.5=h7f8727e_1" >> dependencies.txt
echo "zlib=1.2.12=h7f8727e_2" >> dependencies.txt
```

You can copy and paste this content into a Markdown file, and then add the file to your GitHub repository to document the dependencies of your Conda environment. Make sure to update the file whenever there are changes to the dependencies.

To create this content in the command prompt, you can use the following command:

```bash
echo "# Conda Environment Dependencies" > dependencies.md
echo "Below are the dependencies for the Conda environment named \"py39\":" >> dependencies.md
echo "```" >> dependencies.md
cat packages.txt >> dependencies.md
echo "```" >> dependencies.md
```

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

## License and Usage

The Molecular Similarity Finder script is open-source and provided under a permissive license, granting users the freedom to use and modify it as needed. For comprehensive details regarding its usage and licensing terms, please refer to the script's header section.

---

**Important Note:** Prior to executing the script, it is advisable to thoroughly review its content and tailor it to suit your specific research or application. Detailed instructions and configuration options are comprehensively documented within the script itself.


# CVAE for SMILES Molecular Structures Generation (cvae_0.03.py)

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


## Usage

- The model is trained on a dataset of SMILES strings.
- It can generate new molecular structures based on the learned latent space.

## Dependencies

- PyTorch for model implementation.
- Transformers library for the pre-trained RoBERTa model.
- Other standard Python libraries for data handling and visualization.

## Installation and Execution

- Clone the repository.
- Install required dependencies.
- Run the script with appropriate dataset and parameters.

## Contributing

Contributions to improve the model or extend its functionalities are welcome. Please follow the standard pull request process for contributions.
