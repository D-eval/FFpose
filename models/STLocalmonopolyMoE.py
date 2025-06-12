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
                 num_neighbors,
                 d_in,d_out,
                 time_len,num_layers,
                 loss_threshold=None):
        super().__init__()
        self.loss_threshold = loss_threshold
        self.num_experts = num_experts
        self.all_experts = nn.ModuleList([
            LocalAE(num_neighbors,
                           d_in,d_out,
                           time_len,num_layers) # 没法用 batch_norm
            for e in range(num_experts)
        ])
        self.memory = [0] * num_experts
        
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out
        self.time_len = time_len
        self.num_layers = num_layers
    def add_new_expert(self):
        num_neighbors = self.num_neighbors
        d_in = self.d_in
        d_out = self.d_out
        time_len = self.time_len
        num_layers = self.num_layers
        new_expert = LocalAE(num_neighbors,
                           d_in,d_out,
                           time_len,num_layers)
        self.all_experts.append(new_expert)
        self.num_experts += 1
    def new_expert_pretrain(self,x,e,num_pretrain=700):
        # x: (1,dT,dN,d)
        expert = self.all_experts[e]
        params = expert.parameters()
        optimizer = torch.optim.Adam(params, lr=1e-4)
        for epoch in range(num_pretrain):
            loss, mu_logvar_xhat = expert.get_recon_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("新专家预训练完成:{}".format(loss))
    def everyone_try_this_problem(self, x):
        # (B,dT,dN,d)
        # return: (B,E)
        B = x.shape[0]
        E = self.num_experts
        all_loss = torch.zeros((B,E))
        others = []
        for e,expert in enumerate(self.all_experts):
            losses, mu_logvar_xhat = expert.get_recon_loss(x) # (B,)
            all_loss[:,e] = losses
            others += [mu_logvar_xhat]
        return all_loss, others
    def sparse_others(self,others,expert_idx):
        # others: [(mu,logvar,x_hat) x E] :
        # [((B,D),(B,D),(B,dT,dN,d)) x E]
        # expert_idx: (B,)
        # return: (B,D), (B,D), (B,dT,dN,d)
        all_mu = []
        all_logvar = []
        all_xhat = []
        for b, e in enumerate(expert_idx):
            mu = others[e][0][b] # (D)
            logvar = others[e][1][b] # (D)
            xhat = others[e][2][b] # (dT,dN,d)
            # 储存
            all_mu += [mu]
            all_logvar += [logvar]
            all_xhat += [xhat]
        all_mu = torch.stack(all_mu) # (B,D)
        all_logvar = torch.stack(all_logvar)
        all_xhat = torch.stack(all_xhat) # (B,dT,dN,d)
        return all_mu, all_logvar, all_xhat
    def encode(self, x, expert_idx):
        # x: (B,dT,dN,d)
        # expert_idx: (B,) index of expert
        # return: (B,D),(B,D)
        all_mu = []
        all_logvar = []
        for b, e in enumerate(expert_idx):
            mu, logvar = self.all_experts[e].encode(x[b].unsqueeze(0))
            all_mu += [mu]
            all_logvar += [logvar]
        all_mu = torch.cat(all_mu, dim=0)
        all_logvar = torch.cat(all_logvar, dim=0)
        return all_mu, all_logvar
    def decode(self, z, expert_idx):
        # z: (B,D)
        # expert_idx: (B,)
        # return: (B,dT,dN,d)
        B,D = z.shape
        all_xhat = []
        for b in range(B):
            idx = expert_idx[b]
            expert = self.all_experts[idx]
            xhat = expert.decode(z[b].unsqueeze(0))
            all_xhat += [xhat]
        all_xhat = torch.cat(all_xhat,dim=0)
        return all_xhat
    def get_loss(self, x, kl_weight=0):
        # x: (B,dT,dN,d)
        # (B,D)
        B = x.shape[0]
        all_loss, others = self.everyone_try_this_problem(x) # (B,E)
        # print(all_loss)
        # loss_recons
        losses, expert_idx = torch.min(all_loss, dim=1) # (B,)
        # for b in range(B):
        #     if losses[b]>self.loss_threshold:
        #         x_new = x[b].unsqueeze(0)
        #         self.add_new_expert()
        #         self.new_expert_pretrain(x_new,-1)
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
        losses, mu_logvar_xhat = expert.get_recon_loss(x)
        # (B,)
        loss_recons = losses.mean()
        return loss_recons
    def forward(self, x):
        # x: (B,dT,dN,d)
        # return:(B,D),(B,D),(B,dT,dN,d),(B,)
        all_loss, others = self.everyone_try_this_problem(x)
        losses, expert_idx = torch.min(all_loss, dim=1)
        mu, logvar, x_hat = self.sparse_others(others, expert_idx)
        return mu, logvar, x_hat, expert_idx
    def update_memory(self, expert_idx):
        # (B,)
        for e in expert_idx:
            self.memory[e] += 1
