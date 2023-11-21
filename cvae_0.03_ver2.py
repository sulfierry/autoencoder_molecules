import os
import torch
import pandas as pd
import torch.nn as nn
from rdkit import Chem
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from rdkit.Chem import Descriptors, rdMolDescriptors
from transformers import RobertaModel, RobertaTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
# Configurações básicas
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CPUS = os.cpu_count()
EPOCHS = 1000
BATCH_SIZE = 32
LATENT_DIM = 768
LEARNING_RATE = 1e-3
LOG_INTERVAL = 10
WEIGHT_DECAY = 1e-5 # regularizacao L2

class CVAE(nn.Module):
    def __init__(self, pretrained_model_name, latent_dim, vocab_size, max_sequence_length):
        super(CVAE, self).__init__()
        self.DEVICE = DEVICE
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name)

        self.fc_mu = nn.Linear(self.encoder.config.hidden_size, latent_dim)
        self.fc_var = nn.Linear(self.encoder.config.hidden_size, latent_dim)

        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size

        decoder_output_size = max_sequence_length * vocab_size

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.encoder.config.hidden_size, decoder_output_size),
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
        output = self.decoder(z)
        output = output.view(-1, self.max_sequence_length, self.vocab_size)
        return output

    def forward(self, input_ids, attention_mask):
        mu, log_var = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def train_cvae(cvae, train_dataloader, val_dataloader, test_dataloader, optimizer, num_epochs, log_interval, accumulate_grad_steps=2):
   scaler = GradScaler() # Para precisão mista

   train_losses, val_losses, test_losses = [], [], []

   for epoch in range(num_epochs):
       print(f"Epoch {epoch+1}/{num_epochs}")
       print("Training...")

       # Treino
       cvae.train()
       train_loss = 0
       for batch_idx, (input_ids, attention_mask) in enumerate(train_dataloader):
           input_ids, attention_mask = input_ids.to(cvae.device), attention_mask.to(cvae.device)

           # Inicializa o acumulador de gradiente
           optimizer.zero_grad()

           for i in range(accumulate_grad_steps):
               start_idx = i * len(input_ids)
               end_idx = start_idx + len(input_ids)
               sub_input_ids = input_ids[start_idx:end_idx]
               sub_attention_mask = attention_mask[start_idx:end_idx]

               with autocast():
                  recon_batch, mu, logvar = cvae(sub_input_ids, sub_attention_mask)
                  loss = loss_function(recon_batch, sub_input_ids, mu, logvar) / accumulate_grad_steps

               scaler.scale(loss).backward()

           scaler.step(optimizer)
           scaler.update()

           train_loss += loss.item()

           if batch_idx % log_interval == 0:
               print(f'\tTrain Batch {batch_idx}. Loss: {loss.item() / len(input_ids):.6f}')

       epoch_train_loss = train_loss / len(train_dataloader.dataset)
       train_losses.append(epoch_train_loss)

       # Validação
       cvae.eval()
       val_loss = 0
       with torch.no_grad():
           print("Validating...")
           for input_ids, attention_mask in val_dataloader:
               input_ids, attention_mask = input_ids.to(cvae.device), attention_mask.to(cvae.device)
               recon_batch, mu, logvar = cvae(input_ids, attention_mask)
               val_loss += loss_function(recon_batch, input_ids, mu, logvar).item()

       epoch_val_loss = val_loss / len(val_dataloader.dataset)
       val_losses.append(epoch_val_loss)

       # Teste
       test_loss = 0
       with torch.no_grad():
           print("Testing...")
           for input_ids, attention_mask in test_dataloader:
               input_ids, attention_mask = input_ids.to(cvae.device), attention_mask.to(cvae.device)
               recon_batch, mu, logvar = cvae(input_ids, attention_mask)
               test_loss += loss_function(recon_batch, input_ids, mu, logvar).item()

       epoch_test_loss = test_loss / len(test_dataloader.dataset)
       test_losses.append(epoch_test_loss)

       print(f'Epoch Summary: Train Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}, Test Loss: {epoch_test_loss:.6f}')

   # Plotar o gráfico de perda por época
   plt.figure(figsize=(10, 6))
   plt.plot(train_losses, label='Training Loss')
   plt.plot(val_losses, label='Validation Loss')
   plt.plot(test_losses, label='Test Loss')
   plt.title('Loss vs. Epochs')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.grid(True)
   plt.show()

   return train_losses, val_losses, test_losses


def train_cvae(cvae, train_dataloader, val_dataloader, test_dataloader, optimizer, num_epochs, log_interval, accumulate_grad_steps=4):
    scaler = GradScaler()  # For mixed precision

    # Freezing encoder parameters and enabling grad for decoder and reparameterization layers
    for param in cvae.encoder.parameters():
        param.requires_grad = False

    for param in cvae.decoder.parameters():
        param.requires_grad = True

    for param in cvae.fc_mu.parameters():
        param.requires_grad = True

    for param in cvae.fc_var.parameters():
        param.requires_grad = True

    train_losses, val_losses, test_losses = [], [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("Training...")

        cvae.train()
        train_loss = 0
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        for batch_idx, (input_ids, attention_mask) in enumerate(train_dataloader):
            # Move batch to CUDA
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)

            # Forward pass
            with autocast():
                recon_batch, mu, logvar = cvae(input_ids, attention_mask)
                loss = loss_function(recon_batch, input_ids, mu, logvar) / accumulate_grad_steps

            # Backward pass (accumulate gradients)
            scaler.scale(loss).backward()

            # Update model weights and reset gradients after specified steps
            if (batch_idx + 1) % accumulate_grad_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f'\tTrain Batch {batch_idx}. Loss: {loss.item() / len(input_ids):.6f}')

        epoch_train_loss = train_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        cvae.eval()
        val_loss = 0
        with torch.no_grad():
            print("Validating...")
            for input_ids, attention_mask in val_dataloader:
                input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
                recon_batch, mu, logvar = cvae(input_ids, attention_mask)
                val_loss += loss_function(recon_batch, input_ids, mu, logvar).item()
        
        epoch_val_loss = val_loss / len(val_dataloader.dataset)
        val_losses.append(epoch_val_loss)

        # Testing
        test_loss = 0
        with torch.no_grad():
            print("Testing...")
            for input_ids, attention_mask in test_dataloader:
                input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
                recon_batch, mu, logvar = cvae(input_ids, attention_mask)
                test_loss += loss_function(recon_batch, input_ids, mu, logvar).item()

        epoch_test_loss = test_loss / len(test_dataloader.dataset)
        test_losses.append(epoch_test_loss)

        print(f'Epoch Summary: Train Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}, Test Loss: {epoch_test_loss:.6f}')

    # Plotting the training, validation, and testing losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, val_losses, test_losses


def loss_function(recon_x, x, mu, logvar):
    if recon_x.dim() != 3:
        raise ValueError(f"recon_x should be 3-dimensional, but got shape {recon_x.shape}")

    CE = nn.functional.cross_entropy(recon_x.view(-1, recon_x.size(2)), x.view(-1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return CE + KLD


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
        

def smiles_to_token_ids_parallel(smiles_list, tokenizer):
    with ThreadPoolExecutor(max_workers=NUM_CPUS) as executor:
        futures = [executor.submit(tokenizer, smile, return_tensors='pt', padding=True, truncation=True) for smile in smiles_list]
        results = [future.result() for future in as_completed(futures)]
    return [res['input_ids'] for res in results], [res['attention_mask'] for res in results]


def token_ids_to_smiles(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


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
    dataset = SmilesDataset(smiles_data.tolist(), tokenizer, max_length, num_cpus)

    # Criar o DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def postprocess_smiles(smiles_list, reference_smile):
    reference_mol = Chem.MolFromSmiles(reference_smile)
    reference_properties = calculate_properties(reference_mol)
    processed_smiles = []

    for smile in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                canonical_smile = Chem.MolToSmiles(mol, isomericSmiles=True)
                largest_fragment = max(canonical_smile.split('.'), key=len)
                mol = Chem.MolFromSmiles(largest_fragment)
                
                properties = calculate_properties(mol)
                if is_similar(properties, reference_properties):
                    processed_smiles.append({
                        'smile': largest_fragment,
                        **properties
                    })
            else:
                processed_smiles.append({'smile': 'Invalid SMILES'})
        except Exception as e:
            processed_smiles.append({'smile': f"Error: {str(e)}"})

    return processed_smiles
    
def generate_molecule(cvae, z, tokenizer, method='sampling', top_k=50):
    cvae.eval()
    with torch.no_grad():
        recon_smiles_logits = cvae.decode(z)
        
        if recon_smiles_logits.dim() != 3 or recon_smiles_logits.shape[1] != cvae.max_sequence_length:
            raise ValueError(f"Dimension mismatch in logits: {recon_smiles_logits.shape}")

        if method == 'argmax':
            recon_smiles = torch.argmax(recon_smiles_logits, dim=2)
        elif method == 'sampling':
            probabilities = torch.nn.functional.softmax(recon_smiles_logits, dim=-1)
            recon_smiles = torch.multinomial(probabilities.view(-1, cvae.vocab_size), 1)
            recon_smiles = recon_smiles.view(-1, cvae.max_sequence_length)

        recon_smiles_decoded = token_ids_to_smiles(recon_smiles[0], tokenizer)
        return recon_smiles_decoded

def calculate_properties(mol):
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': rdMolDescriptors.CalcNumHBD(mol),
        'hba': rdMolDescriptors.CalcNumHBA(mol)
    }

def is_similar(prop1, prop2, threshold=0.2):
    for key in prop1:
        if abs(prop1[key] - prop2[key]) > threshold:
            return False
    return True


def main():
    print("Carregando dados...")
    # Carregar dados
    chembl_file_path = './filtered_chembl33_IC50_Kd_ki.tsv'
    chembl_data = pd.read_csv(chembl_file_path, sep='\t')
    smiles_data = chembl_data['canonical_smiles']

    # Parâmetros
    pretrained_model_name = 'seyonec/ChemBERTa-zinc-base-v1'
    batch_size = 32
    max_length = 512
    num_epochs = 10
    log_interval = 100

    # Divisão dos dados em conjuntos de treino, validação e teste
    train_data, test_data = train_test_split(smiles_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

    # Preparar DataLoaders
    print("Preparando DataLoaders...")
    train_dataloader = data_pre_processing(train_data, pretrained_model_name, batch_size, max_length, NUM_CPUS)
    val_dataloader = data_pre_processing(val_data, pretrained_model_name, batch_size, max_length, NUM_CPUS)
    test_dataloader = data_pre_processing(test_data, pretrained_model_name, batch_size, max_length, NUM_CPUS)

    # Inicializar o modelo CVAE
    latent_dim = 768
    vocab_size = 50265
    cvae_model = CVAE(pretrained_model_name, latent_dim, vocab_size, max_length)

    # Inicializar o otimizador
    optimizer = torch.optim.Adam(cvae_model.parameters(), lr=LEARNING_RATE)

    # Treinar o modelo
    print("Iniciando treinamento...")
    train_losses, val_losses, test_losses = train_cvae(cvae_model, train_dataloader, val_dataloader, test_dataloader, optimizer, num_epochs, log_interval)

    print("Treinamento concluído. Salvando o modelo...")
    model_save_path = "./cvae_model.pth"
    torch.save(cvae_model.state_dict(), model_save_path)
    print(f"Modelo salvo em {model_save_path}")

if __name__ == "__main__":
    main()

