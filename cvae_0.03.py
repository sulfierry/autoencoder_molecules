from transformers import RobertaTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset

class SmilesDataset(Dataset):
    """ Dataset personalizado para armazenar e tokenizar dados SMILES. """

    def __init__(self, smiles_list, tokenizer, max_length, num_cpus):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles_list = self.parallel_tokenization(smiles_list, num_cpus)

    def parallel_tokenization(self, smiles_list, num_cpus):
        """ Tokeniza SMILES em paralelo para acelerar o processamento. """
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(self.tokenizer, smile, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length) for smile in smiles_list]
            return [future.result() for future in as_completed(futures)]

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.smiles_list[idx]['input_ids'].squeeze(0), self.smiles_list[idx]['attention_mask'].squeeze(0)


def data_pre_processing(smiles_data, pretrained_model_name, batch_size, max_length, num_cpus):
    """
    Prepara os dados SMILES para treinamento, validação ou teste.

    Args:
        smiles_data (pd.Series): Série Pandas contendo dados SMILES.
        pretrained_model_name (str): Nome do modelo pré-treinado para o tokenizador.
        batch_size (int): Tamanho do lote para o DataLoader.
        max_length (int): Comprimento máximo da sequência de SMILES.
        num_cpus (int): Número de CPUs a serem usadas para tokenização paralela.

    Returns:
        DataLoader: DataLoader pronto para ser usado no treinamento/validação/teste.
    """
    # Carregar o tokenizador
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)

    # Criar o dataset personalizado
    dataset = SmilesDataset(smiles_data, tokenizer, max_length, num_cpus)

    # Criar o DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

import torch.nn.functional as F


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Calcula a função de perda para um CVAE.

    Args:
        recon_x (torch.Tensor): Tensor de saída do decoder do CVAE.
        x (torch.Tensor): Tensor de entrada original.
        mu (torch.Tensor): Média do espaço latente.
        logvar (torch.Tensor): Logaritmo da variância do espaço latente.
        beta (float): Fator de balanceamento entre CE e KLD.

    Returns:
        torch.Tensor: Valor da perda combinada.
    """
    # Verifica as dimensões de recon_x
    if recon_x.dim() != 3:
        raise ValueError(f"recon_x should be 3-dimensional (batch_size, sequence_length, num_classes), but got shape {recon_x.shape}")

    # Entropia Cruzada: compara os SMILES reconstruídos com os originais
    CE = F.cross_entropy(recon_x.view(-1, recon_x.size(2)), x.view(-1), reduction='sum')

    # Divergência KL: regularização do espaço latente
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Perda total combinada
    return CE + beta * KLD

