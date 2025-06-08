import torch
from torch import nn
import torch.nn.functional as F
import math

from abc import ABC, abstractmethod

# fps = 1
# T = 10 # (10秒)

# 对 (T,N,d) 进行聚类, dvae聚类
class KmeansEncoder(nn.Module):
    def __init__(self,T,N,d,C,
                 use_discrete=True):
        super().__init__()
        self.C = C
        self.convs = nn.Conv3d(1,C,(T,N,d),padding=(T//2-1,0,0))
        self.centers = nn.Parameter(torch.randn(C,T,N,d)) # 类中心
        self.use_discrete = use_discrete
    def encode_continue(self,x):
        # x: (B,1,T,N,d)
        B,_,T,N,d = x.shape
        D = T*N*d
        y = self.convs(x)
        # y: (B,C,T,1,1)
        cls_logits, _ = torch.max(y, dim=2) # (B,C,1,1)
        cls_logits = cls_logits[:,:,0,0] # (B,C)
        cls_logits = cls_logits / math.sqrt(D) # (B,C)
        cls_prob = F.softmax(cls_logits,dim=-1) # (B,C) 或者用Gumbel Max
        return cls_prob
    def encode_discrete(self,x):
        # x: (B,1,T,N,d)
        B,_,T,N,d = x.shape
        D = T*N*d
        y = self.convs(x)
        # y: (B,C,T,1,1)
        y = y[:,:,:,0,0] # (B,C,T)
        cls_logits, offsets = torch.max(y, dim=2) # (B,C), (B,C)
        _, cls = torch.max(cls_logits, dim=1) # (B,)
        offset = []
        for b in range(B):
            offset.append(offsets[b,cls[b]]) # (int)
        offset = torch.stack(offset) # (B,int)
        return cls, offsets
    def decode_discrete(self,z):
        # z: (B,)
        centers = self.centers[z,...] # (B,T,N,d)
        return centers