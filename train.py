
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

