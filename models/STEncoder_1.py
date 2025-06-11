import torch
from torch import nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from .MLP import MLP

# batchnorm 核心出装
# >>> x.view(-1,x.shape[-1]).shape
# torch.Size([240, 3])
# >>> x.view(-1,x.shape[-1]).view(*x.shape).shape
# torch.Size([8, 3, 5, 2, 3])

# 局部形状重构
# (B,dT,dN,d) -> (B,D) -> (B,dT,dN,d)


class LocalSTEncoder(nn.Module):
    def __init__(self,joint_idx,
                 num_neighbors,
                 d_in,
                 d_out,
                 time_len=3,
                 num_layers=3):
        super().__init__()
        self.joint_idx = joint_idx
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out
        self.time_len = time_len
        # 邻接关节和自身时间
        dim_in = d_in * ((num_neighbors-1) + time_len)
        self.proj = MLP(dim_in, d_out*2,
                        num_layers, 
                        "linear", 0.1, False)
        self.lin_mean = nn.Linear(d_out*2, d_out)
        self.lin_log_var = nn.Linear(d_out*2, d_out)
    def forward(self,x_tem,x_spa):
        # x_spa: (B,dN,d)
        # x_tem: (B,dT,d)
        # return: z: (B,D)
        B,dN,d = x_spa.shape
        B,dT,d = x_tem.shape
        assert (dT,dN,d)==(self.time_len,self.num_neighbors,self.d_in)
        x_st = torch.cat([x_spa[:,1:],x_tem], dim=1)
        x = torch.flatten(x_st,1,-1) # (B,-1)
        h = self.proj(x)
        mu = self.lin_mean(h)
        log_var = self.lin_log_var(h)
        return mu, log_var


class LocalSTDecoder(nn.Module):
    def __init__(self,joint_idx,
                 num_neighbors,
                 d_latent,
                 d_recons,
                 time_len=3,
                 num_layers=3):
        super().__init__()
        self.joint_idx = joint_idx
        self.num_neighbors = num_neighbors
        self.d_latent = d_latent
        self.d_recons = d_recons
        self.time_len = time_len
        # 三维卷积
        dim_out = d_recons * ((num_neighbors-1) + time_len)
        self.proj = MLP(d_latent, dim_out,
                        num_layers, 
                        "linear", 0.1, False)
        self.dim_out = dim_out
    def forward(self,z):
        # z: (B,D)
        # return: 
        # x_spa: (B,dN,d), 
        # x_tem: (B,dT,d)
        B,D = z.shape
        assert (D)==(self.d_latent)
        dT,dN,d = self.time_len,self.num_neighbors,self.d_recons
        t_center = dT // 2
        x = self.proj(z) # (B,dim_out)
        x = torch.reshape(x,(B,-1,d))
        assert x.shape[1]==(dN-1+dT)
        x_spa = x[:,:(dN-1),:]
        x_tem = x[:,(dN-1):,:]
        x_joint = x_tem[:,t_center,:][:,None,:]
        x_spa = torch.cat([x_joint,x_spa],dim=1)
        # (B,dN,d)
        return x_tem, x_spa

class LocalAE(nn.Module):
    def __init__(self,
                 joint_idx,
                 num_neighbors,
                 d_in,
                 d_out,
                 time_len=3,
                 num_layers=3):
        super().__init__()
        self.encoder = LocalSTEncoder(joint_idx,
                                    num_neighbors,
                                    d_in,
                                    d_out,
                                    time_len,
                                    num_layers)
        self.decoder = LocalSTDecoder(joint_idx,
                                    num_neighbors,
                                    d_latent=d_out,
                                    d_recons=d_in,
                                    time_len=time_len,
                                    num_layers=num_layers)
        self.input_shape = (time_len, num_neighbors, d_in)
        self.latent_dim = d_out
        
    def x_to_spa_tem(self,x):
        # x: (B,dT,dN,d)
        B,dT,dN,d = x.shape
        assert (dT,dN,d)==self.input_shape
        center_idx = dT // 2
        x_tem = x[:,:,0,:] # (B,dT,d)
        x_spa = x[:,center_idx,:,:] # (B,dN,d)
        return x_tem, x_spa

    def encode(self, x):
        # x: (B,dT,dN,d)
        x_tem,x_spa = self.x_to_spa_tem(x)
        mu, logvar = self.encoder(x_tem,x_spa)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: (B,D)
        x_tem, x_spa = self.decoder(z)
        return x_tem, x_spa

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_tem_hat, x_spa_hat = self.decode(z)
        return (x_tem_hat, x_spa_hat), mu, logvar

    def get_recon_loss(self, x):
        # x: (B,dT,dN,d)
        x_tem, x_spa = self.x_to_spa_tem(x)
        # x_spa: (B,dN,d)
        mu, logvar = self.encode(x)
        x_tem_hat, x_spa_hat = self.decode(mu)
        l2_spa = ((x_spa - x_spa_hat)**2).sum(dim=-1).mean(-1)
        l2_tem = ((x_tem - x_tem_hat)**2).sum(dim=-1).mean(-1) # (B,)
        l2 = l2_tem + l2_spa
        return l2, (mu,logvar,(x_tem_hat, x_spa_hat))


def cal_kl_div(mu, logvar):
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div