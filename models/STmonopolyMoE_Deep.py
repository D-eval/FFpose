import torch
from torch import nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

# 局部形状重构
# (B,dT,dN,d) -> (B,D) -> (B,dT,dN,d)

from .STmonopolyMoE import GlobalmonopolyMoE

from monopolyMoE import MonoMoE

def cal_kl_div(mu, logvar):
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div


class DeepME(nn.Module):
    def __init__(self,group_joint_dict,
                 group2_group_dict,
                 group_num_experts,
                 d_in,d_out,
                 time_len_merge,mlp_layers,
                 num_experts_second,
                 d_final):
        super().__init__()
        self.layer1 = GlobalmonopolyMoE(group_joint_dict=group_joint_dict,
                                        group_num_experts=group_num_experts,
                                        d_in=d_in,d_out=d_out,
                                        time_len_merge=time_len_merge,
                                        mlp_layers=mlp_layers)
        group_num = len(group_joint_dict.keys())
        self.group_num = group_num
        self.D1 = d_out
        d_latent = d_out * group_num
        self.layer2 = MonoMoE(num_experts_second,
                              d_latent, d_final,
                              mlp_layers)
        self.D2 = d_final
        self.group2_group_dict = group2_group_dict
        assert len(group2_group_dict)==1
    def load_layers(self, state_dict):
        self.layer1.load_state_dict(state_dict)
    def freeze_layers(self):
        for param in list(self.layer1.parameters()):
            param.requires_grad = False
        self.layer1.eval()
        print("已经冻结层layer1")
    def get_train_params(self):
        params = self.layer2.parameters()
        return params
    def trans_dict_to_Tensor(self, z):
        # z: {G:(B,D)}
        # return: (B,G,D)
        all_z = []
        all_group = self.group2_group_dict["all"]
        for g in all_group:
            all_z += z[g]
        # [(B,D) x G]
        z = torch.stack(all_z,dim=1) # (B,G,D)
        return z
    def trans_Tensor_to_dict(self, z):
        # z: (B,G,D)
        # return: { G:(B,D) }
        z_dict = {}
        all_group = self.group2_group_dict["all"]
        for idx, g in enumerate(all_group):
            z_dict[g] = z[:,idx,:]
        return z_dict
    def forward(self, x):
        # x: (B,T,N,d)
        # return:
        # z1, z1_hat: (B,G*D)
        # z2, logvar2: (B,D2)
        # e1: { G:(B,) }
        # e2: (B,)
        # losses: (B,)
        z1_dict, _, _, g1_e = self.layer1(x)
        z1 = self.trans_dict_to_Tensor(z1_dict) # (B,G,D)
        B,G,D = z1.shape
        z1 = torch.flatten(z1,1,2) # (B,G*D)
        z2, logvar2, z1_hat, e2, losses = self.layer2(z1)
        # z2: (B,D2)
        e1 = g1_e
        return z1, z2, logvar2, z1_hat, e1, e2, losses
    def get_loss(self, x, kl_weight=0):
        # x: (B,T,N,d)
        # return:
        # loss: float
        # e_id: (B,)
        z1, mu, logvar, z1_hat, e_id, _, losses = self(x)
        kl_div = cal_kl_div(mu, logvar)
        recons_loss = losses.mean()
        loss = recons_loss + kl_weight * kl_div
        return loss, e_id
    def decode(self, z2, e2, e1):
        # z2: (B,D2)
        # e2: (B,): int
        # e1: { G:(B,) }: int
        # return:
        # xhat: (B,T,N,d)
        B,D2 = z2.shape
        assert D2==self.D2
        G = self.group_num
        D1 = self.D1
        z1 = self.layer2.decode(z2, e2) # (B,G*D1)
        z1 = torch.reshape(z1, (B,G,D1))
        z1_dict = self.trans_Tensor_to_dict(z1)
        xhat = self.layer1(z1_dict, e1)
        return xhat, z1_dict

