
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae import VariationalAutoEncoder, cal_kl_div

# encoderDecoder合体型的
class MonoMoE(nn.Module):
    def __init__(self, num_experts,
                 input_dim, latent_dim, num_mlp_layers):
        super().__init__()
        self.num_experts = num_experts
        self.all_experts = nn.ModuleList([
            VariationalAutoEncoder(input_dim, latent_dim, num_mlp_layers,
                                   use_bn=False) # 没法用 batch_norm
            for n in range(num_experts)
        ])
        self.memory = [0] * num_experts
    def everyone_try_this_problem(self, x):
        # (B,D)
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
    def sparse_others(self,others, expert_idx):
        # others: [(mu,logvar,x_hat) x E] : [((B,D),(B,D),(B,d)) x E]
        # expert_idx: (B,)
        # return: (B,D), (B,D), (B,d)
        all_mu = []
        all_logvar = []
        all_xhat = []
        for b, e in enumerate(expert_idx):
            mu = others[e][0][b] # (D)
            logvar = others[e][1][b] # (D)
            xhat = others[e][2][b] # (d)
            # 储存
            all_mu += [mu]
            all_logvar += [logvar]
            all_xhat += [xhat]
        all_mu = torch.stack(all_mu) # (B,D)
        all_logvar = torch.stack(all_logvar)
        all_xhat = torch.stack(all_xhat)
        return all_mu, all_logvar, all_xhat
    def encode(self, x, expert_idx):
        # x: (B,d)
        # expert_idx: (B,) index of expert
        # return: (B,D)
        all_mu = []
        all_logvar = []
        for b, e in enumerate(expert_idx):
            mu, logvar = self.all_experts[e].encode(x[b].unsqueeze(0))
            all_mu += [mu]
            all_logvar += [logvar]
        all_mu = torch.cat(all_mu, dim=0)
        all_logvar = torch.cat(all_logvar, dim=0)
        return all_mu, all_logvar

    def get_loss(self, x, kl_weight=0):
        # (B,D)
        all_loss, others = self.everyone_try_this_problem(x) # (B,E)
        # loss_recons
        losses, expert_idx = torch.min(all_loss, dim=1) # (B,)
        loss_recons = losses.mean()
        # loss_kl
        mu, logvar, _ = self.sparse_others(others, expert_idx)
        loss_kl = cal_kl_div(mu, logvar)
        loss = loss_recons + kl_weight * loss_kl
        return loss, expert_idx
    def forward(self, x):
        # x: (B,d)
        # forward是编码
        all_loss, others = self.everyone_try_this_problem(x)
        losses, expert_idx = torch.min(all_loss, dim=1)
        mu, logvar, xhat = self.sparse_others(others, expert_idx)
        return mu, logvar, xhat

    def update_memory(self, expert_idx):
        # (B,)
        for e in expert_idx:
            self.memory[e] += 1
