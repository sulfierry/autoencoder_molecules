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
