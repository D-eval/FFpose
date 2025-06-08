import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from MLP import MLP
import random

def get_a_freeze_mlp(z_dim, x_dim, num_layers, mode):
    transform = MLP(d_in=z_dim, d_out=x_dim, num_layers=num_layers, mode=mode)
    transform.eval()
    for param in transform.parameters():
        param.requires_grad = False
    return transform


# ========== Synthetic Dataset ==========
class DummyClassDataset(Dataset):
    def __init__(self, n_samples=10000, z_dim=8, x_dim=54, transform_fn=None, num_mlp_layers=3,
                 cls_num=2):
        self.n_samples = n_samples
        self.z_dim = z_dim
        self.x_dim = x_dim

        # 生成规则
        torch.manual_seed(42)
        
        # generate z ~ N(μ, σ^2)
        self.mu = torch.randn(n_samples, z_dim)
        self.logvar = torch.randn(n_samples, z_dim) * 0.1  # smaller variance
        z = self.mu + torch.randn_like(self.mu) * torch.exp(0.5 * self.logvar)
        self.z = z
        # 划分边界
        all_boundary = [0]
        step = n_samples // cls_num
        for c in range(1,cls_num):
            boundary = step * c
            all_boundary += [boundary]
        all_boundary += [n_samples]
        self.all_boundary = all_boundary
        # 所有变换
        self.transformes = [
            get_a_freeze_mlp(z_dim, x_dim, num_layers=num_mlp_layers, mode="linear")
            for c in range(cls_num)
        ]
        # 所有样本表象
        xs = []
        labels = []
        for i in range(cls_num):
            zi = z[ all_boundary[i]:all_boundary[i+1] ]
            xi = self.transformes[i](zi)
            xs += [xi]
            labels += xi.shape[0] * [i]
        
        
        self.x = torch.cat(xs,dim=0)
        self.labels = labels
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.x[idx]
        label = self.labels[idx]
        return x, label

    def get_a_rand_data(self):
        idx = random.randint(0,len(self)-1)
        return self[idx]
    
    def get_cls_idx(self, c):
        start = self.all_boundary[c]
        end = self.all_boundary[c+1]
        return start, end