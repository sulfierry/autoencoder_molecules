import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2D representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3D reconstruction
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Extract atom coordinates from a PDB file
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

# Get the atom coordinates from your PDB file
atom_coordinates = extract_atom_coordinates("./3c9t.pdb")

# Calculate the geometric center
geometric_center = np.mean(atom_coordinates, axis=0)

# Define the grid boundaries
grid_half_size = 7.5  # half of 15
grid_min = geometric_center - grid_half_size
grid_max = geometric_center + grid_half_size

# Filter atoms that are inside the grid
atoms_in_grid = atom_coordinates[(atom_coordinates[:, 0] >= grid_min[0]) & (atom_coordinates[:, 0] <= grid_max[0]) &
                                (atom_coordinates[:, 1] >= grid_min[1]) & (atom_coordinates[:, 1] <= grid_max[1]) &
                                (atom_coordinates[:, 2] >= grid_min[2]) & (atom_coordinates[:, 2] <= grid_max[2])]


# Convert the 3D atom coordinates to PyTorch tensors
atoms_tensor = torch.tensor(atoms_in_grid, dtype=torch.float32)

# Create a DataLoader for training
dataset = TensorDataset(atoms_tensor, atoms_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the autoencoder and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 500
losses = []

for epoch in range(num_epochs):
    for data in dataloader:
        inputs, targets = data
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    losses.append(loss.item())

# Plot the training loss
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()

# Pass the original 3D coordinates through the autoencoder
reconstructed_coords = model(atoms_tensor).detach().numpy()

# Calculate the reconstruction error (MSE)
mse = np.mean((atoms_in_grid - reconstructed_coords) ** 2)
print(f"Reconstruction MSE: {mse}")

