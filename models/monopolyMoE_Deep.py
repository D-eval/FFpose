
import torch
import torch.nn as nn
import torch.nn.functional as F

from monopolyMoE import MonoMoE


class DeepMonoMoE(nn.Module):
    def __init__(self, layers_experts,
                 layers_dim, num_mlp_layers):
        super().__init__()
        assert (len(layers_experts)+1)==len(layers_dim)
        self.all_layers = nn.ModuleList([
            MonoMoE(layers_experts[l],
                    layers_dim[l],
                    layers_dim[l+1],
                    num_mlp_layers)
            for l in range(len(layers_experts))
        ])
        self.num_layers = len(layers_experts)
    def get_loss(self, x, deep, kl_weight=0.01):
        # x: (B, d)
        # deep: int :0 ~ self.num_layers
        # 第几层表示? 0为重构当前
        assert deep<self.num_layers
        with torch.no_grad():
            for l in range(deep):
                x, _, _ = self.all_layers[l](x)
        loss, expert_idx = self.all_layers[deep].get_loss(x, kl_weight)
        return loss, expert_idx
    def forward(self, x, deep):
        # (B,d)
        assert deep<self.num_layers
        if deep==0:
            return x, None, None
        x_next = x
        for l in range(deep):
            x_next, logvar, x_hat = self.all_layers[l](x_next)
        mu = x_next
        return mu, logvar, x_hat

