import os
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm.auto import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from concurrent.futures import ThreadPoolExecutor, as_completed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CPUS = os.cpu_count()
EPOCHS = 5000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

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


def smiles_to_token_ids_parallel(smiles_list, tokenizer):

    with ThreadPoolExecutor(max_workers=NUM_CPUS) as executor:
        futures = [executor.submit(tokenizer, smile, return_tensors='pt', padding=True, truncation=True) for smile in smiles_list]
        results = [future.result() for future in as_completed(futures)]
    return [res['input_ids'] for res in results], [res['attention_mask'] for res in results]


# Função para converter token IDs em SMILES
def token_ids_to_smiles(token_ids, tokenizer):
    return tokenizer.decode(token_ids[0], skip_special_tokens=True)

def train_cvae(cvae, dataloader, optimizer, num_epochs, tokenizer, log_interval):
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
    epoch_losses = []  # Lista para armazenar a perda de cada época

    # Inicializar DataFrame para armazenar dados de perda
    loss_data = pd.DataFrame(columns=['Epoch', 'Loss'])

    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            batch_size = input_ids.size(0)
            input_ids, attention_mask = input_ids.to(cvae.DEVICE), attention_mask.to(cvae.DEVICE)

            optimizer.zero_grad()

            # Usando precisão mista
            with autocast():
                recon_batch, mu, logvar = cvae(input_ids, attention_mask)
                loss = loss_function(recon_batch, input_ids, mu, logvar)

            # Backpropagation com ajuste fino
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        epoch_loss = train_loss / len(dataloader.dataset)
        epoch_losses.append(epoch_loss)

        # Atualizar DataFrame com novos dados
        new_row = pd.DataFrame({'Epoch': [epoch], 'Loss': [epoch_loss]})
        loss_data = pd.concat([loss_data, new_row], ignore_index=True)

        print(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}')

    # Salvar dados de perda em um arquivo CSV
    loss_data.to_csv('loss_data.csv', index=False)

    # Plotar o gráfico de perda por época
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label='Training Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('loss_epochs.png')
    plt.close()

    return epoch_losses


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

class CVAE(nn.Module):
    def __init__(self, pretrained_model_name, latent_dim, vocab_size, max_sequence_length):
        super(CVAE, self).__init__()
        self.DEVICE = DEVICE
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name)

        self.fc_mu = nn.Linear(self.encoder.config.hidden_size, latent_dim)
        self.fc_var = nn.Linear(self.encoder.config.hidden_size, latent_dim)

        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size

        # Tamanho de saída para o decodificador
        decoder_output_size = max_sequence_length * vocab_size
        print(f"Decoder output size: {decoder_output_size}")

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.encoder.config.hidden_size, decoder_output_size),
            nn.Unflatten(1, (max_sequence_length, vocab_size)),
            nn.LogSoftmax(dim=-1)
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_states, dim=1)
        mu = self.fc_mu(pooled_output)
        log_var = self.fc_var(pooled_output)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        output = self.decoder[0:3](z)
       # print("Shape before unflatten:", output.shape)
        # Remover uma dimensão extra se ela existir
        if output.dim() > 2:
            output = output.squeeze(1)
            print("Shape after squeeze:", output.shape)
        if output.shape[-1] != self.max_sequence_length * self.vocab_size:
            raise RuntimeError(f"Incorrect shape before unflatten. Got {output.shape}, expected last dimension to be {self.max_sequence_length * self.vocab_size}")
        return self.decoder[3:](output)


    def forward(self, input_ids, attention_mask):
        mu, log_var = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# Certifique-se de que todas as classes e funções necessárias estejam importadas ou definidas aqui.
# Isso inclui CVAE, SmilesDataset, train_cvae, smiles_to_token_ids, generate_molecule.
def main(smiles_input, pretrained_model_name, pkidb_file_path, num_epochs=EPOCHS, batch_size=BATCH_SIZE):
    
    start_time = time.time()
    
    # Tokenizador e modelo pré-treinado são carregados
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
    vocab_size = tokenizer.vocab_size

    # Instancia o CVAE com o modelo pré-treinado, dimensão latente e tamanho do vocabulário
    cvae = CVAE(pretrained_model_name=pretrained_model_name, 
                latent_dim=256, 
                vocab_size=vocab_size, 
                max_sequence_length=tokenizer.model_max_length).to(DEVICE)

    # Prepara o dataset e o dataloader
    dataset = SmilesDataset(pkidb_file_path, tokenizer, max_length=tokenizer.model_max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_CPUS)

    # Configura o otimizador
    optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)
    #optimizer = optim.RMSprop(cvae.parameters(), lr=1e-3)
    #optimizer = optim.NAdam(cvae.parameters(), lr=1e-3)
    #optimizer = optim.AdamW(cvae.parameters(), lr=1e-3)

    # Treina o CVAE
    train_cvae(cvae, dataloader, optimizer, num_epochs, tokenizer, log_interval=10)

    # Gera uma nova molécula
    input_ids, attention_mask = smiles_to_token_ids_parallel(smiles_input, tokenizer)
    input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
    z = cvae.encode(input_ids, attention_mask)[0]  # Obtém apenas o mu (média) do espaço latente
    z = z.unsqueeze(0)  # Simula um lote de tamanho 1 para compatibilidade de formato
    generated_smile = generate_molecule(cvae, z, tokenizer)
    
    # Salvar o estado do dicionário do modelo
    torch.save(cvae.state_dict(), 'cvae_finetuned.pth')
	
    print(f"Generated SMILES: {generated_smile}")

    # Parar o cronômetro e imprimir o tempo total
    end_time = time.time()
    print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")   
 
    # Liberação de memória da GPU
    del cvae
    torch.cuda.empty_cache()

# Se este script estiver sendo executado como o script principal, execute a função main.
if __name__ == '__main__':
    smiles_input = 'C1=CC(=CC=C1NC(=O)C[C@@H](C(=O)O)N)OC2=CC(=C(C=C2Br)F)F'
    pretrained_model_name = 'seyonec/ChemBERTa-zinc-base-v1'
    pkidb_file_path =  './pkidb_2023-06-30.tsv'  # Atualize para o caminho correto
    main(smiles_input, pretrained_model_name, pkidb_file_path)