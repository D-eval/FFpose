import torch
from torch import nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

# 局部形状重构
# (B,dT,dN,d) -> (B,D) -> (B,dT,dN,d)



class ForwardForwardModule(nn.Module,ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def normless_forward(self, *args, **kwargs):
        """必须实现：无 layer norm 的前向传播"""
        pass
    @abstractmethod
    def get_loss(self, *args, **kwargs):
        """必须实现：局部 loss 计算"""
        pass


# layer1: 
class Layer1(ForwardForwardModule):
    def __init__(self,d,D,joint_num,kernel_size):
        # kernel_size=5, fps=10, 每个卷积核感受0.5秒
        super().__init__()
        # 寻找0.5秒内的D种模式, N,d维度 share kernel
        self.D = D
        self.kernel_size = kernel_size
        self.kernel = nn.Parameter(torch.randn(D, 1, kernel_size) * 0.1)
    def apply_conv1d(self, x):
        # x: (B*N*d, 1, T)
        return F.conv1d(x, self.kernel, padding=self.kernel.shape[-1] // 2)  # (B*N*d, D, T)
    def normless_forward(self,x):
        # x: (B,T,N,d)
        # return: (B,T1,N,d,D)
        # 不经过norm的forward
        B,T,N,d = x.shape
        D = self.D
        x = x.permute(0,2,3,1) # (B,N,d,T)
        x = torch.flatten(x,0,2) # (B*N*d,T)
        x = x.unsqueeze(1) # (B*N*d,1,T)
        y = self.apply_conv1d(x) # (B*N*d,D,T)
        z = F.relu(y)
        z = F.max_pool1d(z, kernel_size=self.kernel_size) # (B*N*d,D,T//2) T1=T//2
        # 把 B*N*d 个样本分成 B 类
        T1 = z.shape[-1]
        z = torch.reshape(z,(B,N,d,D,T1)) # (B,N,d,D,T1)
        z = z.permute(0,4,1,2,3) # (B,T1,N,d,D)
        return z
    def get_loss(self, x, label_one_hot):
        # x: (B,T,N,d)
        # label_one_hot: (B,C)
        exist_cls = (label_one_hot.sum(0) > 0) # (C,)
        assert exist_cls.sum() >= 2, "至少需要两个类"
        B,T,N,d = x.shape
        C = label_one_hot.shape[1]
        D = self.D
        # 统计量
        z = self.normless_forward(x) # (B,T1,N,d,D)
        T1 = z.shape[1]
        all_cls_num = label_one_hot.sum(dim=0) + 1e-9 # (C,)
        center = (label_one_hot.T @ z) # (C,T1,N,d,D)
        center += (~exist_cls) * 1 # batch中没有的类心放在远处 1 / 1e-9 = 1e9, (3*1e9**2)**(1/2)
        center /= all_cls_num[:,None,None,None,None] # (C,T1,N,d,D) 不存在的center在0
        center_for_z = label_one_hot @ center # (B,T1,N,d,D)
        dist_to_others = (z[:,None,:,:,:,:] - center[None,:,:,:,:,:]) # (B,C,T1,N,d,D)
        dist_to_others = (dist_to_others**2).sum(-1).sqrt() # (B,C,T1,N,d)
        dist_to_others *= (~label_one_hot[:,:,None,None,None]) + 1e-9
        loss_others = -torch.log(dist_to_others).mean() # 非常大的时候梯度为0
        dist_to_center = ((z - center_for_z)**2).sum(-1).sqrt() # (B,T1,N,d)
        loss_center = dist_to_center.mean()
        loss = loss_others + loss_center
        return loss
    def get_params(self):
        return [self.kernel]
    def forward(self, x):
        # x: (B,T,N,d)
        # return: (B,T1,N,d,D)
        B,T,N,d = x.shape
        D = self.D
        z = self.normless_forward(x) # (B,T1,N,d,D)
        T1 = z.shape[1]
        # 对D归一化, 忽略距离信息
        w = torch.softmax(z,dim=1) # (B,T1,N,d,D)
        return w

# 空间识别
class Layer2(ForwardForwardModule):
    def __init__(self,num_joints,d,D):
        super().__init__()
        # (B,T,N*d)
        d_in = num_joints * d
        self.linear = nn.Linear(d_in, D)
        
    def normless_forward(self, x):
        # x: (B,T,N,d)
        B,T,N,d = x.shape
        x = torch.flatten(x,2,3) # (B,T,N*d)
        y = self.linear(x)
        z = self.relu(y)
        return z # (B,T,D)
    



# 空间的抽象表示和时间抽象表示应当相似。