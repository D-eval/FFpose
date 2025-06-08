
import torch
import torch.nn as nn
import torch.nn.functional as F

from MLP import MLP
from vae import MuSigmaEncoder

# 分开型的
class MonoMoE(nn.Module):
    def __init__(self, num_encoder, num_decoder,
                 input_dim, latent_dim, num_mlp_layers):
        super().__init__()
        self.num_encoder = num_encoder
        self.all_encoders = nn.ModuleList([
            MuSigmaEncoder(input_dim, latent_dim, num_mlp_layers)
            for n in range(num_encoder)
        ])

        self.num_decoder = num_decoder
        self.all_decoders = nn.ModuleList([
            MLP(latent_dim, input_dim, num_layers=num_mlp_layers, mode="linear")
            for n in range(num_encoder)
        ])
