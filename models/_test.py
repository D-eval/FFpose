import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from MLP import MLP

# ----------- AutoEncoder -----------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = MLP(input_dim, latent_dim, num_layers=3, mode="kernel_3")
        self.decoder = MLP(latent_dim, input_dim, num_layers=3, mode="kernel_3")

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# ----------- 数据生成（两个高斯簇） -----------
torch.manual_seed(0)
N = 1000
data1 = torch.randn(N, 2) * 0.8 + torch.tensor([2.0, 2.0])
data2 = torch.randn(N, 2) * 0.8 + torch.tensor([-2.0, -2.0])
data = torch.cat([data1, data2], dim=0)

dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# ----------- 模型训练 -----------
model = AutoEncoder(input_dim=2, latent_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for (x,) in dataloader:
        x_hat = model(x)
        loss = loss_fn(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch+1:02d}: Loss = {total_loss / len(dataset):.4f}")

# ----------- 可视化 -----------
model.eval()
with torch.no_grad():
    x_recon = model(data)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Data")
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)

plt.subplot(1, 2, 2)
plt.title("Reconstructed Data")
plt.scatter(x_recon[:, 0], x_recon[:, 1], alpha=0.5, color='orange')

plt.tight_layout()
plt.savefig("./show.pdf")
plt.close()

