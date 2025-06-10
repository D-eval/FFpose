import torch
from torch import nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

# 局部形状重构
# (B,dT,dN,d) -> (B,D) -> (B,dT,dN,d)

from .STLocalmonopolyMoE import LocalMonopolyMoE

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
    def get_j_loss(self,x,j, kl_weight=0):
        # x: (B,dT,N,d)
        # t,j: int
        B,dT,N,d = x.shape
        assert self.time_len_merge==dT
        neighbors = self.joint_neighbor_dict[j]
        dt_half = self.time_len_merge // 2
        dx = x[:,:,neighbors,:]
        joint_monopoly = self.localMoE[str(j)]
        loss, expert_idx = joint_monopoly.get_loss(dx,kl_weight=kl_weight)
        return loss, expert_idx
    def get_loss(self,x,kl_weight=0):
        # x: (B,dT,N,d)
        B,dT,N,d = x.shape
        assert dT==self.time_len_merge
        total_loss = 0
        for j in range(self.joint_num):
            loss, expert_idx = self.get_j_loss(x,j,kl_weight)
            total_loss += loss
        total_loss = total_loss / self.joint_num
        # 返回最后一个 expert_idx
        return total_loss, expert_idx

