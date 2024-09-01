import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE


# Load Data

df = pd.read_csv('/preprocessed_insurance_data.csv')
X = df.to_numpy(dtype=np.float32)
X_tensor = torch.tensor(X)
dataset = TensorDataset(X_tensor)
original_data_loader = DataLoader(dataset, batch_size=256, shuffle=True) 
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, final=False):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(True)
        )
        decoder_layers = [
            nn.Linear(output_dim, input_dim),
            nn.ReLU(True)  # Default to ReLU
        ]

        if final:
            # Change the last activation to sigmoid only for the final layer
            decoder_layers[-1] = nn.Sigmoid()
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, noise_factor=0.1):
        x_noisy = x + noise_factor * torch.randn_like(x)
        x_noisy = torch.clamp(x_noisy, 0., 1.)
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        """ This function returns the encoded output without adding noise or decoding. """
        return self.encoder(x)


def train_autoencoder_layer(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    losses = []  # List to store loss for each epoch
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs in data_loader:  # Unpack the inputs directly
            inputs = inputs[0]  # DataLoader wraps each batch in a list, so you need to access the first element
            optimizer.zero_grad()
            reconstructed, _ = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}')
    return model, losses

# Define dimensions for each layer
dims = [95, 50, 25, 10, 5, 2]
models = []
losses_per_layer = []  # List to store losses for each layer

for i in range(len(dims) - 1):
    model = DenoisingAutoencoder(dims[i], dims[i + 1], final=(i == len(dims) - 2))
    optimizer = optim.Adam(model.parameters(), lr=0.0025)
    criterion = nn.MSELoss()
    trained_model, losses = train_autoencoder_layer(model, data_loader, criterion, optimizer, num_epochs=30)
    models.append(trained_model)
    losses_per_layer.append(losses)
    # Update data_loader for the next layer's input
    with torch.no_grad():
        new_X = []
        for inputs in data_loader:
            inputs = inputs[0]  # DataLoader wraps each batch in a list, so you need to access the first element
            _, encoded = model(inputs)
            new_X.append(encoded)
        new_X = torch.cat(new_X, dim=0)
        data_loader = DataLoader(TensorDataset(new_X, new_X), batch_size=256, shuffle=True)

# Plotting
plt.figure(figsize=(10, 5))
for i, loss_layer in enumerate(losses_per_layer):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_layer) + 1), loss_layer)
    plt.title(f'Training Loss Across Epochs (Layer {i+1})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


print("\n End Of AutoEncoder")
print("\n Start Of Clustering \n")


def extract_features(data_loader, models):
    current_loader = data_loader
    for model in models:
        new_features = []
        for batch in current_loader:
            inputs = batch[0]
            with torch.no_grad():
                _, encoded = model(inputs)
                new_features.append(encoded)
        # Update current_loader with the new features for the next layer
        new_features = torch.cat(new_features, dim=0)
        current_loader = DataLoader(TensorDataset(new_features), batch_size=256, shuffle=False)
    return new_features

# Use the original data loader that contains the raw data
encoded_features = extract_features(original_data_loader, models)


# Function to compute soft assignments using Student's t-distribution
def soft_assignment(features, cluster_centers, alpha=1.0):
    distance_squared = torch.sum((features.unsqueeze(1) - cluster_centers) ** 2, dim=2)
    inverse_distances = 1.0 / (1.0 + distance_squared / alpha)
    weights = (inverse_distances.t() / torch.sum(inverse_distances, dim=1)).t()
    return weights

# Function to calculate the target distribution from the soft assignments
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# Initialize cluster centers with k-means
kmeans = KMeans(n_clusters=5, n_init=10, random_state=0).fit(encoded_features)
cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, requires_grad=True)

# Prepare the DataLoader with encoded features
encoded_tensor = encoded_features.detach().clone()
data_loader = DataLoader(TensorDataset(encoded_tensor), batch_size=256, shuffle=True)

# DEC training loop using SGD as per the paper's methodology
optimizer = torch.optim.SGD([cluster_centers], lr=0.001, momentum=0.9)
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        
        q = soft_assignment(inputs, cluster_centers)
        p = target_distribution(q)
        
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}')


# Evaluation Metrics
cluster_labels = kmeans.labels_
score = silhouette_score(encoded_features, cluster_labels)
db_index = davies_bouldin_score(encoded_features, cluster_labels)
ch_index = calinski_harabasz_score(encoded_features, cluster_labels)

print("\n Model Performance:\n")
print(f"Silhouette Score: {score}")
print(f"Davies-Bouldin Index: {db_index}")
print(f"Calinski-Harabasz Index: {ch_index}")


# Initialize and fit t-SNE on the encoded features
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(encoded_features)

# Plotting the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('Cluster Visualization with t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()


df['Cluster'] = cluster_labels

