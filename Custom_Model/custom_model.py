import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE


df = pd.read_csv('/preprocessed_insurance_data.csv')
X = df.to_numpy()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(95, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, 5),
            nn.ELU(inplace=True),
            nn.Linear(5, 2)  # Compressed representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ELU(inplace=True),
            nn.Linear(5, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 95),
            nn.Sigmoid()  # Ensuring output is in the range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.003)

def train_autoencoder(model, criterion, optimizer, data_loader, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        for batch in data_loader:  # Changed 'data' to 'batch' for clarity
            inputs = batch[0]  # Ensure inputs is a tensor
            optimizer.zero_grad()
            recon = model(inputs)  # Make sure inputs is correctly passed to the model
            loss = criterion(recon, inputs)  # Compare reconstruction against the original inputs
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return losses
        
# Assume X is your dataset
X_tensor = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

losses = train_autoencoder(autoencoder, criterion, optimizer, data_loader, num_epochs=100)

print("\n")

def calculate_reconstruction_error(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    return average_loss

# Calculate average reconstruction error on training dataset
train_recon_error = calculate_reconstruction_error(autoencoder, data_loader)
print(f'Average Reconstruction Error on Training Data: {train_recon_error}')

def calculate_MSE(model, data_loader):
    model.eval()
    total_mse = 0
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0]
            outputs = model(inputs)
            mse = nn.MSELoss()(outputs, inputs)
            total_mse += mse.item()
            count += 1
    average_mse = total_mse / count
    print(f'Average MSE: {average_mse}')
    
calculate_MSE(autoencoder, data_loader)

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss Across Epochs (AutoEncoder)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("\n End Of AutoEncoder")
print("\n Start Of Clustering \n")

def extract_features(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0]
            encoded_features = model.encoder(inputs)
            features.append(encoded_features.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

# Extract features from the autoencoder
encoded_features = extract_features(autoencoder, data_loader)
n_clusters = 5
kmeans = KMeans(n_clusters, random_state=0, n_init=10).fit(encoded_features)  # Explicitly set n_init to avoid warning
cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)


def soft_assignment(features, cluster_centers):
    # Student's t-distribution kernel
    norm_squared = torch.sum((features.unsqueeze(1) - cluster_centers) ** 2, dim=2)
    weights = 1 / (1 + norm_squared)
    weights = (weights.t() / torch.sum(weights, dim=1)).t()  # Normalize over all clusters
    return weights

def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()

def clustering_loss(q, p):
    p = p.clamp(min=1e-10)  # To prevent log(0)
    return torch.sum(q * torch.log(q / p))

def combined_loss(data, recon, q, p):
    recon_loss = nn.MSELoss()(recon, data)
    kl_loss = clustering_loss(q, p)
    return recon_loss + kl_loss

num_epochs = 10
losses = []  # List to store loss values

for epoch in range(num_epochs):
    epoch_losses = []  # List to store losses of each batch in the current epoch
    for batch in data_loader:
        inputs = batch[0]
        optimizer.zero_grad()

        outputs = autoencoder(inputs)
        features = autoencoder.encoder(inputs)

        q = soft_assignment(features, cluster_centers)
        p = target_distribution(q)

        loss = combined_loss(inputs, outputs, q, p)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    # Calculate average loss for the epoch
    average_loss = (sum(epoch_losses) / len(epoch_losses)) / 100
    losses.append(average_loss)
    print(f'Epoch {epoch+1}, Average Loss: {average_loss}')

# Plotting the convergence
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), losses, label='Average Training Loss per Epoch')
plt.title('Training Loss Across Epochs (K-Means)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("\n Model Performance:\n")

cluster_labels = kmeans.labels_

score = silhouette_score(encoded_features, cluster_labels)
print(f"Silhouette Score: {score}\n")

db_index = davies_bouldin_score(encoded_features, cluster_labels)
print(f"Davies-Bouldin Index: {db_index}\n")

ch_index = calinski_harabasz_score(encoded_features, cluster_labels)
print(f"Calinski-Harabasz Index: {ch_index}\n")


# Initialize and fit t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(encoded_features)

# Plotting
plt.figure(figsize=(10, 8))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.colorbar()
plt.title('Cluster Visualization with t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()


df['Cluster'] = cluster_labels
df.to_csv('/clustered.csv', index=False)

