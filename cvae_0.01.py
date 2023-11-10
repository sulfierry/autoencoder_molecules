import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm.auto import tqdm
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel


# Assumindo que as classes e funções auxiliares foram definidas conforme o esboço anterior...
# Aqui, vamos definir o Dataset que carregará os dados do PKIDB e preparará os inputs para o CVAE
# Correções na classe SmilesDataset
class SmilesDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = pd.read_csv(file_path, sep='\t')
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile = self.data.iloc[idx]['Canonical_Smiles']
        inputs = self.tokenizer(smile, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

def loss_function(recon_x, x, mu, logvar):
    # Verifique se recon_x tem três dimensões. Se não, algo está errado.
    if recon_x.dim() != 3:
        raise ValueError(f"recon_x should be 3-dimensional (batch_size, sequence_length, num_classes), but got shape {recon_x.shape}")

    # Flatten os logits e os índices de classe para passar para a cross_entropy.
    # A função cross_entropy espera logits no formato (batch_size * sequence_length, num_classes)
    # e índices de classe no formato (batch_size * sequence_length).
    CE = nn.functional.cross_entropy(recon_x.view(-1, recon_x.size(2)), x.view(-1), reduction='sum')

    # Cálculo da Divergência KL
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return CE + KLD


# Função para converter SMILES em token IDs
def smiles_to_token_ids(smiles, tokenizer):
    tokens = tokenizer(smiles, return_tensors='pt', padding=True, truncation=True)
    return tokens['input_ids'], tokens['attention_mask']

# Função para converter token IDs em SMILES
def token_ids_to_smiles(token_ids, tokenizer):
    return tokenizer.decode(token_ids[0], skip_special_tokens=True)
