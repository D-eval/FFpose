import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from MLP import MLP
import random

from itertools import product
# 笛卡尔积

def get_a_freeze_mlp(z_dim, x_dim, num_layers, mode):
    transform = MLP(d_in=z_dim, d_out=x_dim, num_layers=num_layers, mode=mode)
    transform.eval()
    for param in transform.parameters():
        param.requires_grad = False
    return transform


# ========== Synthetic Dataset ==========
class HierarchyDataset(Dataset):
    def __init__(self, n_each_cls=1000,
                 num_mlp_layers=3,
                 layers_cls=[2,3],
                 layers_dim=[8,16,54]):
        self.n_each_cls = n_each_cls
        z_dim = layers_dim[0]
        x_dim = layers_dim[-1]
        self.z_dim = z_dim
        self.x_dim = x_dim
        # 生成规则
        torch.manual_seed(42)

        assert (len(layers_cls)+1)==len(layers_dim)
        self.transforms = [
            [get_a_freeze_mlp(layers_dim[i], layers_dim[i+1],
                              num_layers=num_mlp_layers, mode="linear")
             for c in range(layers_cls[i])]
            for i in range(len(layers_cls))
        ] # 不必使用ModuleList
        
        # generate z ~ N(μ, σ^2)
        z = torch.randn(layers_cls + [n_each_cls, z_dim])
        x = torch.zeros(layers_cls + [n_each_cls, x_dim])
        all_cls_pathway = layers_cls
        ranges = [range(n) for n in layers_cls]
        index_tuples = list(product(*ranges))
        self.index_tuples = index_tuples

        for idxs in index_tuples:
            h = z[idxs]
            for l in range(len(layers_cls)):
                mlp = self.transforms[l][idxs[l]]
                h = mlp(h)
            x[idxs] = h
        self.x = x

        n_samples = n_each_cls
        for c in layers_cls:
            n_samples *= c
        self.n_samples = n_samples

        # 展平成标准形式 (N, dim)
        self.x = x.reshape(-1, self.x_dim)
        self.z = z.reshape(-1, self.z_dim)
        
        self.labels = []
        for path in product(*[range(n) for n in layers_cls]):
            for i in range(n_each_cls):
                self.labels.append(path + (i,))  # 完整索引
        self.labels = torch.tensor([full_idx[:-1] for full_idx in self.labels])
                
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.x[idx]
        label = self.labels[idx]
        return x, label

    def get_a_rand_data(self):
        idx = random.randint(0,len(self)-1)
        return self[idx]
