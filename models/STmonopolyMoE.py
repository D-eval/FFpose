import torch
from torch import nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

# 局部形状重构
# (B,dT,dN,d) -> (B,D) -> (B,dT,dN,d)

from STLocalmonopolyMoE import LocalMonopolyMoE

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
            joint: LocalMonopolyMoE(num_experts,
                                    joint,
                                    len(joint_neighbor_dict[joint]),
                                    d_in,d_out,
                                    time_len_merge,
                                    mlp_layers)
            for joint in joint_neighbor_dict.keys()
        })
        self.time_len_merge = time_len_merge
    def get_t_j_loss(self,x,t,j, kl_weight=0):
        # x: (B,T,N,d)
        # t,j: int
        neighbors = self.joint_neighbor_dict[j]
        dt_half = self.time_len_merge // 2
        dx = x[:,(t-dt_half):(t+dt_half+1),neighbors,:]
        joint_monopoly = self.localMoE[j]
        loss, expert_idx = joint_monopoly.get_loss(dx,kl_weight=kl_weight)
        return loss, expert_idx

