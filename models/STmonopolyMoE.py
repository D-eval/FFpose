import torch
from torch import nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

# 局部形状重构
# (B,dT,dN,d) -> (B,D) -> (B,dT,dN,d)

from .STLocalmonopolyMoE import LocalMonopolyMoE

class GlobalmonopolyMoE(nn.Module):
    def __init__(self,
                 group_joint_dict,
                 group_num_experts,
                 d_in,d_out,
                 time_len_merge,
                 mlp_layers,
                 loss_threshold=None):
        assert time_len_merge%2==1
        super().__init__()
        
        joint_num = 0
        for k,v in group_joint_dict.items():
            joint_num += len(v)
        self.joint_num = joint_num
        
        self.group_num = len(group_joint_dict.keys())
        
        self.d_in = d_in
        
        self.group_joint_dict = group_joint_dict
        self.localMoE = nn.ModuleDict({
            group: LocalMonopolyMoE(group_num_experts[group],
                                    len(group_joint_dict[group]),
                                    d_in,d_out,
                                    time_len_merge,
                                    mlp_layers,
                                    loss_threshold)
            for group in group_joint_dict.keys()
        })
        self.group_num = len(group_joint_dict.keys())
        self.time_len_merge = time_len_merge
        self.group_num_experts = group_num_experts
    def get_g_loss(self,x, g, kl_weight=0):
        # x: (B,dT,N,d)
        # g: nameOfGroup, str
        B,dT,N,d = x.shape
        assert self.time_len_merge==dT
        joints = self.group_joint_dict[g]
        dx = x[:,:,joints,:]
        local_monopoly = self.localMoE[g]
        loss, expert_idx = local_monopoly.get_loss(dx,kl_weight=kl_weight)
        return loss, expert_idx
    def get_loss(self,x,kl_weight=0):
        # x: (B,dT,N,d)
        # return:
        # group_expert_idx: {g:e}
        B,dT,N,d = x.shape
        assert dT==self.time_len_merge
        total_loss = 0
        group_expert_idx = {}
        for g in self.group_joint_dict.keys():
            loss, expert_idx = self.get_g_loss(x,g,kl_weight)
            total_loss += loss
            group_expert_idx[g] = expert_idx
        total_loss = total_loss / self.group_num
        # (J,B) -> (B,J)
        return total_loss, group_expert_idx
    def forward_g(self,x,g):
        # x: (B,dT,N,d)
        # j: int
        # return:
        # return:(B,D),(B,D),(B,dT,d)
        B,dT,N,d = x.shape
        assert self.time_len_merge==dT
        neighbors = self.group_joint_dict[g]
        dx = x[:,:,neighbors,:] # (B,dT,dN,d)
        joint_monopoly = self.localMoE[g]
        mu, logvar, x_hat, expert_idx = joint_monopoly(dx)
        return mu, logvar, x_hat, expert_idx
    def local_dict_to_global_xhat(self,group_xhat):
        # group_xhat: { g: (B,dT,dN,d) }
        
        group_joint = self.group_joint_dict
        
        B = list(group_xhat.values())[0].shape[0]
        N = self.joint_num
        T = self.time_len_merge
        d = self.d_in
        
        device = list(group_xhat.values())[0].device
        
        xhat = torch.zeros((B,T,N,d)).to(device)
        for g in group_xhat.keys():
            joint_idx = group_joint[g]
            xhat_local = group_xhat[g]
            xhat[:,:,joint_idx,:] = xhat_local
        return xhat
    def forward(self, x):
        # x: (B,dT,N,d)
        # return:
        # mu: { g:(B,D) }
        # logvar: { g:(B,D) }
        # xhat: { g:(B,N(g),d) }
        # e_index: { g:(B,) }
        B,dT,N,d = x.shape
        assert dT==self.time_len_merge
        group_expert_idx = {}
        group_mu = {}
        group_logvar = {}
        group_xhat = {}
        for g in self.group_joint_dict.keys():
            # (B,D), (B,D), (B,dT,dN,d), (B,)
            mu, logvar, x_hat, expert_idx = self.forward_g(x,g)
            group_expert_idx[g] = expert_idx
            group_mu[g] = mu
            group_logvar[g] = logvar
            group_xhat[g] = x_hat
        out_xhat = self.local_dict_to_global_xhat(group_xhat)
        return group_mu, group_logvar, out_xhat, group_expert_idx
    def decode_g(self,z,g,expert_idx):
        # z: (B,D)
        # expert_idx: (B,)
        local_MoE = self.localMoE[g]
        x_hat = local_MoE.decode(z, expert_idx)
        return x_hat
    def decode(self,z,expert_idx):
        # z: { G:(B,D) }
        # expert_idx: { G:(B,) }
        x_dict = {}
        for g in z.keys():
            local_z = z[g] # (B,D)
            local_expert_idx = expert_idx[g]
            local_xhat = self.decode_g(local_z,g,local_expert_idx)
            x_dict[g] = local_xhat
        xhat = self.local_dict_to_global_xhat(x_dict)
        return xhat
    def merge_cluster(self,expert_idx,
                      merge_group=["arm_R","leg_L","waist"]):
        # expert_idx: { G:(B,) }
        # return: (B,)
        assert self.group_num_experts["arm_R"]==5
        B = expert_idx["arm_R"].shape[0]
        weigths = {}
        w = 1
        for g in merge_group:
            weigths[g] = w
            w *= 5
        merged_cluster = torch.zeros((B,))
        for g in merge_group:
            merged_cluster += expert_idx[g] * weigths[g]
        return merged_cluster.to(torch.long)
    # 以下是 pretrain function
    def soldier_step_out(self,x,g,e):
        # x: (B,dT,N,d)
        # g: str, name of group
        # e: int, index of expert
        B,dT,N,d = x.shape
        assert self.time_len_merge==dT
        neighbors = self.group_joint_dict[g]
        dx = x[:,:,neighbors,:]
        joint_monopoly = self.localMoE[g]
        loss = joint_monopoly.soldier_step_out(dx,e)
        return loss
    def team_step_out(self,x,e):
        # x: (B,dT,N,d)
        # e: int, index of expert
        B,dT,N,d = x.shape
        assert dT==self.time_len_merge
        total_loss = 0
        for g in self.group_joint_dict.keys():
            loss = self.soldier_step_out(x,g,e)
            total_loss += loss
        total_loss = total_loss / self.group_num
        # 返回最后一个 expert_idx
        return total_loss
    def get_e_parameters(self,e):
        # e: int, index of expert
        params = []
        for g in self.group_joint_dict.keys():
            params+=list(self.localMoE[g].all_experts[e].parameters())
        return params


