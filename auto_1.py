import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_atom_coordinates(pdb_path):
    atom_coords = []
    with open(pdb_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_coords.append([x, y, z])
    return np.array(atom_coords)

def augment_data(atom_coordinates, num_augmentations=5):
    augmented_data = [atom_coordinates]
    for _ in range(num_augmentations):
        noise = np.random.normal(0, 0.5, atom_coordinates.shape)
        augmented_data.append(atom_coordinates + noise)
    return np.vstack(augmented_data)

class ImprovedAutoencoder(nn.Module):
    def __init__(self):
        super(ImprovedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

atom_coordinates = extract_atom_coordinates("./3c9t.pdb")
geometric_center = np.mean(atom_coordinates, axis=0)
grid_half_size = 7.5
grid_min = geometric_center - grid_half_size
grid_max = geometric_center + grid_half_size
atoms_in_grid = atom_coordinates[
    (atom_coordinates[:, 0] >= grid_min[0]) & (atom_coordinates[:, 0] <= grid_max[0]) &
    (atom_coordinates[:, 1] >= grid_min[1]) & (atom_coordinates[:, 1] <= grid_max[1]) &
    (atom_coordinates[:, 2] >= grid_min[2]) & (atom_coordinates[:, 2] <= grid_max[2])
]

# Normalize the data
mean = atoms_in_grid.mean(axis=0)
std = atoms_in_grid.std(axis=0)
atoms_in_grid = (atoms_in_grid - mean) / std

augmented_atoms = augment_data(atoms_in_grid)
augmented_atoms_tensor = torch.tensor(augmented_atoms, dtype=torch.float32)

dataset = TensorDataset(augmented_atoms_tensor, augmented_atoms_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ImprovedAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training process
num_epochs = 1000
train_losses = []
val_losses = []
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, targets = data
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            running_val_loss += val_loss.item()
    val_losses.append(running_val_loss / len(val_loader))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

atoms_tensor = torch.tensor(atoms_in_grid, dtype=torch.float32)
dataset = TensorDataset(atoms_tensor, atoms_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

encoded_2d = model.encoder(atoms_tensor).detach().numpy()
plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('2D Representation of 3D Structures')
plt.grid(True)
plt.show()




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Passe as coordenadas 3D originais pelo autoencoder
encoded_2d = model.encoder(atoms_tensor).detach().numpy()
reconstructed_3d = model.decoder(torch.tensor(encoded_2d, dtype=torch.float32)).detach().numpy()

# Crie um gráfico 3D para as coordenadas originais e reconstruídas
fig = plt.figure(figsize=(12, 6))

# Coordenadas 3D originais
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(atoms_in_grid[:, 0], atoms_in_grid[:, 1], atoms_in_grid[:, 2], c='b', marker='o')
ax1.set_title('Original 3D Coordinates')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Coordenadas 3D reconstruídas
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(reconstructed_3d[:, 0], reconstructed_3d[:, 1], reconstructed_3d[:, 2], c='r', marker='o')
ax2.set_title('Reconstructed 3D Coordinates')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.show()


def rmsd(original, reconstructed):
    diff = original - reconstructed
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

rms_value = rmsd(atoms_in_grid, reconstructed_3d)
print(f"RMSD between the original and reconstructed coordinates: {rms_value:.4f} Å")


import matplotlib.pyplot as plt

# 1. Histograma de Erros
errors = np.linalg.norm(atoms_in_grid - reconstructed_3d, axis=1)
plt.hist(errors, bins=50, alpha=0.75, color='blue')
plt.xlabel('Erro')
plt.ylabel('Número de Átomos')
plt.title('Histograma de Erros de Reconstrução')
plt.grid(True)
plt.show()

# 2. Salvar os PDBs
def save_to_pdb(coords, filename):
    with open(filename, 'w') as f:
        for i, coord in enumerate(coords):
            f.write("ATOM  {:5d}  CA  ALA A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C  \n".format(
                i+1, i+1, coord[0], coord[1], coord[2]))
        f.write("END\n")

save_to_pdb(atoms_in_grid, 'original.pdb')
save_to_pdb(reconstructed_3d, 'reconstructed.pdb')

# 3. Análise de Resíduos
residues = np.linalg.norm(atoms_in_grid - reconstructed_3d, axis=1)
original_norms = np.linalg.norm(atoms_in_grid, axis=1)

plt.scatter(original_norms, residues, alpha=0.5, color='red')
plt.xlabel('Magnitude das Coordenadas Originais')
plt.ylabel('Resíduos')
plt.title('Análise de Resíduos')
plt.grid(True)
plt.show()
