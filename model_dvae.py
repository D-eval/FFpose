import torch
from torch import nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

# clip, bert, MoE, forward-forward
# spectral norm

# 用无监督学习的技巧，把关节表示为离散token并还原。
# 仅需encoder就可以搞定

# 自动编码器
# 一维离散表示、二维离散表示
# 1 (B,T,N,d) -> (B,T,D1) {1,...,M} -> (B,T,N,d)
# 2 (B,T,D1) -> (B,D2,D2) {1,...,M} -> (B,T,D1)

# 样本(B,T) 特征(N,d)
# (B,T,N*d) -linear-> (B,T,D) 我们希望加入一些连续性
# (B,N*d,T) -conv1d-> (B,D,T) (N*d * D) * K
# (B,N,d)(T) : 把(B,N,d)当作一个样本，包含T个特征
# (B,T,N,d) -> (B,T,N,d,1) -> (B*N*d,1,T) -> (B*N*d,D,T) -> (B,T,N,d,D+1)

