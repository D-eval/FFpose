
import torch

from read_data import AMASSDataset


device = torch.device("cuda")

data_path = ["/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/CMU"]
train_set = AMASSDataset(data_path,
                         device=device,
                         time_len=9,
                         use_hand=False,
                         target_fps=1,
                         use_6d=True)

all_cls = []
for i in range(len(train_set)):
    x, label = train_set[i]
    if label in all_cls:
        continue
    else:
        all_cls.append(label)

 