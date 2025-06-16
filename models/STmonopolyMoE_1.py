import torch
from torch import nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

# 局部形状重构
# (B,dT,dN,d) -> (B,D) -> (B,dT,dN,d)

from .STLocalmonopolyMoE_1 import LocalMonopolyMoE

class GlobalmonopolyMoE(nn.Module):
    def __init__(self,
                 joint_neighbor_dict,
                 num_experts,
                 d_in,d_out,
                 time_len_merge,
                 mlp_layers):
        assert time_len_merge%2==1
        super().__init__()
        self.joint_neighbor_dict = joint_neighbor_dict
        self.localMoE = nn.ModuleDict({
            str(joint): LocalMonopolyMoE(num_experts,
                                    joint,
                                    len(joint_neighbor_dict[joint]),
                                    d_in,d_out,
                                    time_len_merge,
                                    mlp_layers)
            for joint in joint_neighbor_dict.keys()
        })
        self.joint_num = len(joint_neighbor_dict.keys())
        self.time_len_merge = time_len_merge
        self.num_experts = num_experts
    def get_j_loss(self,x,j, kl_weight=0):
        # x: (B,dT,N,d)
        # j: int
        B,dT,N,d = x.shape
        assert self.time_len_merge==dT
        neighbors = self.joint_neighbor_dict[j]
        dt_half = self.time_len_merge // 2
        dx = x[:,:,neighbors,:]
        joint_monopoly = self.localMoE[str(j)]
        loss, expert_idx = joint_monopoly.get_loss(dx,kl_weight=kl_weight)
        return loss, expert_idx
    def forward_j(self,x,j):
        # x: (B,dT,N,d)
        # j: int
        # return:
        # return:(B,D),(B,D),(B,dT,d)
        B,dT,N,d = x.shape
        assert self.time_len_merge==dT
        neighbors = self.joint_neighbor_dict[j]
        dt_half = self.time_len_merge // 2
        dx = x[:,:,neighbors,:] # (B,dT,dN,d)
        joint_monopoly = self.localMoE[str(j)]
        mu, logvar, x_hat, expert_idx = joint_monopoly(dx)
        return mu, logvar, x_hat, expert_idx
    def get_loss(self,x,kl_weight=0):
        # x: (B,dT,N,d)
        # return:
        # joint_expert_idx: (B,J,(0)): int
        B,dT,N,d = x.shape
        assert dT==self.time_len_merge
        total_loss = 0
        joint_expert_idx = []
        for j in range(self.joint_num):
            loss, expert_idx = self.get_j_loss(x,j,kl_weight)
            total_loss += loss
            joint_expert_idx += [expert_idx]
        total_loss = total_loss / self.joint_num
        # (J,B) -> (B,J)
        joint_expert_idx = torch.stack(joint_expert_idx) # (J,B)
        joint_expert_idx = joint_expert_idx.transpose(0,1)
        return total_loss, joint_expert_idx
    def forward(self, x):
        # x: (B,dT,N,d)
        # return:
        # mu: (B,N,D)
        # logvar: (B,N,D)
        # xhat: (B,dT,N,d)
        # e_index: (B,N)
        B,dT,N,d = x.shape
        assert dT==self.time_len_merge
        joint_expert_idx = []
        joint_mu = []
        joint_logvar = []
        joint_xhat = []
        for j in range(self.joint_num):
            # (B,D), ~, (B,dT,d), (B,)
            mu, logvar, x_hat, expert_idx = self.forward_j(x,j)
            joint_expert_idx += [expert_idx]
            joint_mu += [mu]
            joint_logvar += [logvar]
            joint_xhat += [x_hat]
        joint_expert_idx = torch.stack(joint_expert_idx) # (J,B)
        joint_mu = torch.stack(joint_mu) # (J,B,D)
        joint_logvar = torch.stack(joint_logvar) # (J,B,D) 
        joint_xhat = torch.stack(joint_xhat) # (J,B,dT,d)
        
        out_xhat = joint_xhat.permute(1,2,0,3)
        out_mu = joint_mu.permute(1,0,2)
        out_logvar = joint_logvar.permute(1,0,2)
        out_e_index = joint_expert_idx.permute(1,0)
        return out_mu, out_logvar, out_xhat, out_e_index
    # 以下是 pretrain function
    def soldier_step_out(self,x,j,e):
        # x: (B,dT,N,d)
        # j: int, index of joint
        # e: int, index of expert
        B,dT,N,d = x.shape
        assert self.time_len_merge==dT
        neighbors = self.joint_neighbor_dict[j]
        dt_half = self.time_len_merge // 2
        dx = x[:,:,neighbors,:]
        joint_monopoly = self.localMoE[str(j)]
        loss = joint_monopoly.soldier_step_out(dx,e)
        return loss
    def team_step_out(self,x,e):
        # x: (B,dT,N,d)
        # e: int, index of expert
        B,dT,N,d = x.shape
        assert dT==self.time_len_merge
        total_loss = 0
        for j in range(self.joint_num):
            loss = self.soldier_step_out(x,j,e)
            total_loss += loss
        total_loss = total_loss / self.joint_num
        # 返回最后一个 expert_idx
        return total_loss
    def get_e_parameters(self,e):
        # e: int, index of expert
        params = []
        for j in range(self.joint_num):
            params+=list(self.localMoE[str(j)].all_experts[e].parameters())
        return params


