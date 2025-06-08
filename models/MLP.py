import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,d_in,
                 d_out,
                 num_layers,
                 mode="linear",
                 p_dropout=0.1,
                 use_bn=True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        assert num_layers>=2, "要考虑表层和输出层"
        self.num_layers = num_layers
        if mode=="linear":
            dims = torch.linspace(d_in,d_out,num_layers).to(torch.long).tolist()
        elif mode.split('_')[0]=="kernel":
            kernel = int(mode.split('_')[1])
            dims = [d_in]
            for i in range(num_layers-2):
                dims += [kernel]
            dims += [d_out]
        else:
            raise NotImplementedError("没写")
        # print(dims)
        self.linears = nn.ModuleList([
            nn.Linear(dims[i],dims[i+1])
            for i in range(num_layers-1)
        ])
        if use_bn:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(dims[i+1])
                for i in range(num_layers-1)
            ])
        self.use_bn = use_bn
        self.dropout = nn.Dropout(p_dropout)
    def forward(self,x):
        # x: (...,d_in)
        # return: (...,d_out)
        d_in = x.shape[-1]
        assert d_in==self.d_in, "输入维度不等，got {}".format(d_in)
        for i in range(self.num_layers-1):
            x = self.linears[i](x)
            if i != self.num_layers-2:
                if self.use_bn:
                    x = self.batch_norms[i](x.view(-1, x.shape[-1])).view(*x.shape)
                x = F.relu(x)
                x = self.dropout(x)
        return x

