import torch
from torch import nn
import torch.nn.functional as F

# 采样率调低，比如 10Hz

# (B,T,N,d)

# 对每个关节，衡量到D个时间曲线距离

# 先线性，再用池化层在时间上扫描，压缩时间，得到曲线的各个部分到D个曲线局部的距离
# (B,T,N,d) -> (N,B,T,d) -> (N,B,T,D) -> (N,B,D,T) -> (N*B*D,1,T) -> (N*B*D,1,T//2)

# forward-forward,不依赖反向传播，每一个nn.Module根据goodness自主更新，有update方法

# 请用自定义卷积核apply卷积
# (N*B*d,1,T) -conv1d-> (N*B*d,D,T) -pool-> (N*B*d,D,T//2)
# (N*B*d,D,T//2) -> (N,B,d,D,T//2) -> (B,N,T//2,d,D) -relu-> (B,N,T//2,d,D) -> (B,N*T//2*d,D), 用于计算goodness
# 统计量:
# label_one_hot: (B,C) -> (C,B)
# center: (C,B) @ (B,N*T//2*d,D) -> (C,N*T//2*d,D) -mean1-> (C,D) 各类的类心。
# 决定更新方向
# 每个样本叠加C速度分量：靠近类心，远离其他类, C个速度分量方向指向或反向指向类心，到本类长度为距离，到其他类长度为距离*1/(C-1)
# 得到更新速度分量 (B,N*T//2*d,C,D) # 在R^D上的C个速度
# 对C个速度分量应用dropout (也就是将相应的D维行向量全部变成0)
# 得到更新速度 (B,N*T//2*d,D)
# 更新速度对卷积核求导，更新卷积核

# goodness的计算涉及pool,只用到stride以内的最大值
# 时间上的相似度尽可能小

# layer1: 
class Layer1(nn.Module):
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

# 空间融合
class SpatialAttention(nn.Module):
    def __init__(self,d,D1,D2,num_heads):
        super().__init__()
        assert D2 % num_heads == 0, "必须有 num_heads 整除 D2"
        self.q = nn.Linear(D1, D2)
        self.k = nn.Linear(D1, D2)
        self.v = nn.Linear(D1, D2)
    def forward(self, x):
        # x: (B,T1,N,d,D1)
        pass