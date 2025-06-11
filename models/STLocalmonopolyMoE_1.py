import torch
from torch import nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

# 局部形状重构
# (B,dT,dN,d) -> (B,D) -> (B,dT,dN,d)

from .STEncoder import LocalAE

def cal_kl_div(mu, logvar):
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div


class LocalMonopolyMoE(nn.Module):
    def __init__(self, num_experts,
                 joint_idx,num_neighbors,
                 d_in,d_out,
                 time_len,num_layers):
        super().__init__()
        self.num_experts = num_experts
        self.all_experts = nn.ModuleList([
            LocalAE(joint_idx,num_neighbors,
                           d_in,d_out,
                           time_len,num_layers) # 没法用 batch_norm
            for e in range(num_experts)
        ])
        self.memory = [0] * num_experts
    def everyone_try_this_problem(self, x):
        # (B,dT,dN,d)
        # return: (B,E)
        B = x.shape[0]
        E = self.num_experts
        all_loss = torch.zeros((B,E))
        others = []
        for e,expert in enumerate(self.all_experts):
            losses, mu_logvar_xtemspahat = expert.get_recon_loss(x) # (B,)
            all_loss[:,e] = losses
            others += [mu_logvar_xtemspahat]
        return all_loss, others
    def sparse_others(self,others,expert_idx):
        # others: [(mu,logvar,x_hat) x E] :
        # [((B,D),(B,D),(B,dT,dN,d)) x E]
        # expert_idx: (B,)
        # return: (B,D), (B,D), (B,dT,d)
        all_mu = []
        all_logvar = []
        all_xhat = []
        for b, e in enumerate(expert_idx):
            mu = others[e][0][b] # (D)
            logvar = others[e][1][b] # (D)
            xhat = others[e][2][0][b] # (dT,d)
            # 储存
            all_mu += [mu]
            all_logvar += [logvar]
            all_xhat += [xhat]
        all_mu = torch.stack(all_mu) # (B,D)
        all_logvar = torch.stack(all_logvar)
        all_xhat = torch.stack(all_xhat) # (B,dT,d)
        return all_mu, all_logvar, all_xhat
    def encode(self, x, expert_idx):
        # x: (B,dT,dN,d)
        # expert_idx: (B,) index of expert
        # return: (B,D)
        all_mu = []
        all_logvar = []
        for b, e in enumerate(expert_idx):
            mu, logvar, x_hat = self.all_experts[e].encode(x[b].unsqueeze(0))
            all_mu += [mu]
            all_logvar += [logvar]
        all_mu = torch.cat(all_mu, dim=0)
        all_logvar = torch.cat(all_logvar, dim=0)
        return all_mu, all_logvar, x_hat
    def get_loss(self, x, kl_weight=0):
        # x: (B,dT,dN,d)
        # (B,D)
        all_loss, others = self.everyone_try_this_problem(x) # (B,E)
        # loss_recons
        losses, expert_idx = torch.min(all_loss, dim=1) # (B,)
        loss_recons = losses.mean()
        # loss_kl
        mu, logvar, x_hat = self.sparse_others(others, expert_idx)
        loss_kl = cal_kl_div(mu, logvar)
        loss = loss_recons + kl_weight * loss_kl
        return loss, expert_idx
    def soldier_step_out(self, x, e):
        # x: (B,dT,dN,d)
        # e: int, index of expert
        expert = self.all_experts[e]
        losses, mu_logvar_xtemspahat = expert.get_recon_loss(x)
        # (B,)
        loss_recons = losses.mean()
        return loss_recons
    def forward(self, x):
        # x: (B,dT,dN,d)
        # return:(B,D),(B,D),(B,dT,d),(B,)
        all_loss, others = self.everyone_try_this_problem(x)
        losses, expert_idx = torch.min(all_loss, dim=1)
        mu, logvar, x_hat = self.sparse_others(others, expert_idx)
        return mu, logvar, x_hat, expert_idx
    def update_memory(self, expert_idx):
        # (B,)
        for e in expert_idx:
            self.memory[e] += 1
