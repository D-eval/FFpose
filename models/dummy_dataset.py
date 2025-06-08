import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from MLP import MLP

# ========== Synthetic Dataset ==========
class FactorizedDataset(Dataset):
    def __init__(self, n_samples=10000, z_dim=8, x_dim=50, transform_fn=None, num_mlp_layers=3):
        self.n_samples = n_samples
        self.z_dim = z_dim
        self.x_dim = x_dim

        # 生成规则
        torch.manual_seed(42)
        
        # Step 1: generate z ~ N(μ, σ^2)
        self.mu = torch.randn(n_samples, z_dim)
        self.logvar = torch.randn(n_samples, z_dim) * 0.1  # smaller variance
        self.z = self.mu + torch.randn_like(self.mu) * torch.exp(0.5 * self.logvar)
        # (N,D)

        self.transform = MLP(z_dim, x_dim, num_layers=num_mlp_layers, mode="linear")
        self.transform.eval()
        for param in self.transform.parameters():
            param.requires_grad = False
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        z = self.z[idx] # (D)
        x = self.transform(z) # (d)
        return x, z
