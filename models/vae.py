
import torch
import torch.nn as nn
import torch.nn.functional as F

from .MLP import MLP

class MuSigmaEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_mlp_layers):
        super().__init__()
        self.encoder = MLP(input_dim, latent_dim, num_layers=num_mlp_layers, mode="linear")
        self.fc_mu = nn.Linear(latent_dim,latent_dim) # MLP(latent_dim, latent_dim, num_layers=2, mode="kernel_{}".format(latent_dim))
        self.fc_logvar = nn.Linear(latent_dim,latent_dim) # MLP(latent_dim, latent_dim, num_layers=2, mode="kernel_{}".format(latent_dim))
    def forward(self, x):
        # (B,d)
        h = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# ----------- AutoEncoder -----------
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_mlp_layers,
                 use_bn = True):
        super().__init__()
        self.encoder = MLP(input_dim, latent_dim, num_layers=num_mlp_layers, mode="linear", use_bn=use_bn)
        self.fc_mu = nn.Linear(latent_dim,latent_dim) # MLP(latent_dim, latent_dim, num_layers=2, mode="kernel_{}".format(latent_dim))
        self.fc_logvar = nn.Linear(latent_dim,latent_dim) # MLP(latent_dim, latent_dim, num_layers=2, mode="kernel_{}".format(latent_dim))
        
        self.decoder = MLP(latent_dim, input_dim, num_layers=num_mlp_layers, mode="linear", use_bn=use_bn)

        self.projector = nn.Linear(latent_dim, 2)  # 降到 2D
        self.recover = nn.Linear(2,latent_dim)
        self.memory_bank = []

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    # 以下均为辅助功能
    def visual_z(self,z):
        # z: (B,D)
        vis_z = self.projector(z)
        return vis_z

    def recons_z(self,z):
        vis_z = self.projector(z)
        z_recons = self.recover(vis_z)
        return z_recons

    def remember(self, x_batch):
        # (B,d_in)
        with torch.no_grad():
            mu, _ = self.encode(x_batch)
            mu_vis = self.visual_z(mu)
            self.memory_bank.append(mu_vis)
            
    def recall(self):
        return self.memory_bank
    
    def get_recon_loss(self, x):
        # x: (B,D)
        mu, logvar = self.encode(x)
        x_hat = self.decode(mu)
        l2 = ((x - x_hat)**2).mean(dim=1) # (B,)
        return l2, (mu,logvar,x_hat)


def cal_kl_div(mu, logvar):
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div

def vae_loss(x_hat, x, mu, logvar, kl_weight=0.01):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div * kl_weight, recon_loss, kl_div


def auxiliary_projection_loss(mu, mu_recon):
    """
    给定 latent mu 和投影-重构的 mu_recon，计算辅助 loss。
    注意：mu.detach() 使 projector 的训练不影响主 Encoder。
    """
    return F.mse_loss(mu_recon, mu.detach())