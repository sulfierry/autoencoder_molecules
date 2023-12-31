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
   dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpus, pin_memory=True)

   return dataloader


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

def train_cvae(cvae, train_dataloader, val_dataloader, test_dataloader, optimizer, num_epochs, log_interval, accumulate_grad_steps=2):
    scaler = GradScaler()  # For mixed precision

    train_losses, val_losses, test_losses = [], [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        cvae.train()
        train_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Process a portion of each batch and accumulate gradients
            for i in range(accumulate_grad_steps):
                start_idx = i * len(input_ids) // accumulate_grad_steps
                end_idx = (i + 1) * len(input_ids) // accumulate_grad_steps
                sub_input_ids = input_ids[start_idx:end_idx].to(cvae.device)
                sub_attention_mask = attention_mask[start_idx:end_idx].to(cvae.device)

                with autocast():
                    recon_batch, mu, logvar = cvae(sub_input_ids, sub_attention_mask)
                    loss = loss_function(recon_batch, sub_input_ids, mu, logvar) / accumulate_grad_steps

                scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(input_ids)}/{len(train_dataloader.dataset)}]\tLoss: {loss.item() / len(sub_input_ids):.6f}')

        # Calculating the training loss for the epoch
        epoch_train_loss = train_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
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

        # Testing
        test_loss = 0
        with torch.no_grad():
            print("Testing...")
            for input_ids, attention_mask in test_dataloader:
                input_ids, attention_mask = input_ids.to(cvae.device), attention_mask.to(cvae.device)
                recon_batch, mu, logvar = cvae(input_ids, attention_mask)
                test_loss += loss_function(recon_batch, input_ids, mu, logvar).item()

        epoch_test_loss = test_loss / len(test_dataloader.dataset)
        test_losses.append(epoch_test_loss)

        # Epoch summary
        print(f'Epoch Summary: Train Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}, Test Loss: {epoch_test_loss:.6f}')

    # Plotting the training, validation, and test losses
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



def token_ids_to_smiles(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

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

class CVAE(nn.Module):
    """Autoencoder Variacional Condicional (CVAE) para manipulação de SMILES."""

    def __init__(self, pretrained_model_name, latent_dim, vocab_size, max_sequence_length, device):
        super(CVAE, self).__init__()
        self.device = device  # Adicionando o dispositivo como um atributo da classe

        # Inicialização do Encoder
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name).to(self.device)

        # Camadas para média e log-variação do espaço latente
        self.fc_mu = nn.Linear(self.encoder.config.hidden_size, latent_dim).to(self.device)
        self.fc_var = nn.Linear(self.encoder.config.hidden_size, latent_dim).to(self.device)

        # Ajuste das dimensões para o Decoder
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        decoder_output_size = max_sequence_length * vocab_size
        intermediate_size = 64  # Tamanho intermediário para a camada do Decoder

        # Construção do Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, decoder_output_size),
            nn.LogSoftmax(dim=-1)
        ).to(self.device)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        last_hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_states, dim=1)
        mu = self.fc_mu(pooled_output)
        log_var = self.fc_var(pooled_output)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def decode(self, z):
        output = self.decoder(z)
        output = output.view(-1, self.max_sequence_length, self.vocab_size)
        return output

    def forward(self, input_ids, attention_mask):
        mu, log_var = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def generate_molecule(self, z, tokenizer, method='sampling', top_k=50):
        self.eval()
        with torch.no_grad():
            recon_smiles_logits = self.decode(z.to(self.device))

            if recon_smiles_logits.dim() != 3 or recon_smiles_logits.shape[1] != self.max_sequence_length:
                raise ValueError(f"Dimension mismatch in logits: {recon_smiles_logits.shape}")

            if method == 'argmax':
                recon_smiles = torch.argmax(recon_smiles_logits, dim=2)
            elif method == 'sampling':
                probabilities = torch.nn.functional.softmax(recon_smiles_logits, dim=-1)
                recon_smiles = torch.multinomial(probabilities.view(-1, self.vocab_size), 1)
                recon_smiles = recon_smiles.view(-1, self.max_sequence_length)

            recon_smiles_decoded = tokenizer.decode(recon_smiles[0], skip_special_tokens=True)
            return recon_smiles_decoded

def main():
    print("Carregando dados...")
    # Defina o caminho para o seu arquivo de dados e carregue-os
    chembl_file_path = './filtered_chembl33_IC50_Kd_ki.tsv'  # Substitua pelo caminho correto
    chembl_data = pd.read_csv(chembl_file_path, sep='\t')
    smiles_data = chembl_data['canonical_smiles']

    
    # Defina parâmetros
    pretrained_model_name = 'seyonec/ChemBERTa-zinc-base-v1'
    batch_size = 32
    max_length = 512
    num_cpus = os.cpu_count()  # Ajuste de acordo com a sua máquina
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Divisão dos dados em treino, validação e teste
    train_data, test_data = train_test_split(smiles_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    # Seleção dos dados de treino, validação e teste
    chembl_data['set'] = 'train'
    chembl_data.loc[test_data.index, 'set'] = 'test'
    chembl_data.loc[val_data.index, 'set'] = 'validation'

    # Preparar DataLoaders
    print("Preparando DataLoaders...")
    num_cpus = os.cpu_count()  # Ajuste este valor conforme adequado para o seu ambiente
    train_dataloader = data_pre_processing(train_data, pretrained_model_name, batch_size, max_length, num_cpus)
    val_dataloader = data_pre_processing(val_data, pretrained_model_name, batch_size, max_length, num_cpus)
    test_dataloader = data_pre_processing(test_data, pretrained_model_name, batch_size, max_length, num_cpus)


    # Inicializar o modelo CVAE
    latent_dim = 768  # Exemplo de dimensão latente
    vocab_size = 50265  # Vocabulário do ChemBERTa, ajuste conforme necessário
    cvae_model = CVAE(pretrained_model_name, latent_dim, vocab_size, max_length, device).to(device)

    # Inicializar o otimizador
    optimizer = torch.optim.Adam(cvae_model.parameters(), lr=1e-4)

    # Treinar o modelo
    print("Iniciando treinamento...")
    num_epochs = 10  # Defina o número de épocas
    log_interval = 100  # Intervalo de log
    train_losses, val_losses, test_losses = train_cvae(cvae_model, train_dataloader, val_dataloader, test_dataloader, optimizer, num_epochs, log_interval)

    print("Treinamento concluído. Salvando o modelo...")
    model_save_path = "./cvae_model.pth"  # Substitua pelo caminho desejado
    torch.save(cvae_model.state_dict(), model_save_path)
    print(f"Modelo salvo em {model_save_path}")
    
if __name__ == "__main__":
    main()
