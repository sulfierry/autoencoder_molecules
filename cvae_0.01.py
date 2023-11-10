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

def train_cvae(cvae, dataloader, optimizer, num_epochs, log_interval):
    # Congelar os parâmetros do encoder, caso você queira fazer ajuste fino apenas do decoder
    for param in cvae.encoder.parameters():
        param.requires_grad = False

    # Ajuste fino apenas dos parâmetros do decoder e da camada de reparametrização
    for param in cvae.decoder.parameters():
        param.requires_grad = True
    for param in cvae.fc_mu.parameters():
        param.requires_grad = True
    for param in cvae.fc_var.parameters():
        param.requires_grad = True

    scaler = GradScaler()  # Inicializa o GradScaler para precisão mista

    cvae.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            batch_size = input_ids.size(0)  # Armazena o tamanho do lote
            input_ids, attention_mask = input_ids.to(cvae.device), attention_mask.to(cvae.device)

            optimizer.zero_grad()

            # Usando precisão mista
            with autocast():
                recon_batch, mu, logvar = cvae(input_ids, attention_mask)
                loss = loss_function(recon_batch, input_ids, mu, logvar)

            # Backpropagation com ajuste fino
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Liberação explícita de memória
            del input_ids, attention_mask, recon_batch, mu, logvar
            torch.cuda.empty_cache()

            train_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * batch_size}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / batch_size:.6f}')

        # Log da perda média após cada época
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')


# Função para gerar moléculas com o modelo
def generate_molecule(cvae, z, tokenizer):
    cvae.eval()
    with torch.no_grad():
        # Assumimos que z é um batch de vetores latentes
        recon_smiles_logits = cvae.decode(z)
        # Aqui precisamos converter logits para SMILES reais, o que pode ser um processo complexo
        # que envolve a escolha do melhor caminho através dos logits. Um exemplo simplificado seria:
        recon_smiles = torch.argmax(recon_smiles_logits, dim=2)
        # Vamos decodificar o primeiro exemplo do batch
        recon_smiles_decoded = token_ids_to_smiles(recon_smiles[0], tokenizer)
        return recon_smiles_decoded
