
import torch

from read_data import AMASSDataset


from models.STmonopolyMoE import GlobalmonopolyMoE
from configs.center_neighbor_idx_dict import center_neighbor_idx_dict_smpl as joint_neighbor_dict

num_experts = 5
d_in = 6
d_out = 64
time_len_merge = 3
mlp_layers = 3


kl_weight = 0.01

device = torch.device("cuda")

data_path = ["/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/CMU"]
train_set = AMASSDataset(data_path,
                         device=device,
                         time_len=1.5,
                         use_hand=False,
                         target_fps=2,
                         use_6d=True)
'''
all_cls = []
for i in range(len(train_set)):
    x, label = train_set[i]
    if label in all_cls:
        continue
    else:
        all_cls.append(label)
'''

monopoly = GlobalmonopolyMoE(joint_neighbor_dict=joint_neighbor_dict,
                             num_experts=num_experts,
                             d_in=d_in,d_out=d_out,
                             time_len_merge=time_len_merge,mlp_layers=mlp_layers)

monopoly.to(device)

x, label = train_set[0]
x = x.unsqueeze(0).to(device)

loss, expert_idx = monopoly.get_loss(x, kl_weight=kl_weight)

# for i in range(23):
#     print(x[:,:,joint_neighbor_dict[i],:].shape)
