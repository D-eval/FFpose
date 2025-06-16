# watch -n 1 nvidia-smi


import torch

from read_data import AMASSDataset


from models.STmonopolyMoE import GlobalmonopolyMoE
from configs.group_joint_dict import group_joint_dict

from torch.utils.data import DataLoader, TensorDataset

from models.eval_cluster import get_nmi_ari_purity, save_confusion_matrix_heatmap

import random

import os

from tqdm import tqdm

def save_bvh(x,file_dir,save_name):
    # x: (T,N,d)
    x_9d = dataset.processor.trans_6d_to_9d(x)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    dataset.processor.write_bvh(x_9d,
                                filename=save_name,
                                save_dir=file_dir)


d_in = 6

time_len_merge = 3
mlp_layers = 3

batch_size = 2048

kl_weight = 0.01

device = torch.device("cuda")

data_path = ["/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/CMU"]

save_dir_general = "../experiment_save/experiment8"


num_pretrain = 500
epochs = 10
lr = 1e-3

threshold_loss = 0.05


weight_kl = 0

need_train = True

num_visual = 10 # 可视化的个数 （写bvh文件）

dataset = AMASSDataset(data_path,
                         device=device,
                         time_len=1.5,
                         use_hand=False,
                         target_fps=2,
                         use_6d=True)

idx_to_show = torch.randint(0,len(dataset)-1,(num_visual,))


# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_name(init_num_experts,d_out,need_pretrain):
    experiment_name = "E_{}_D_{}_pretrain_{}".format(init_num_experts,d_out,need_pretrain)
    return experiment_name


# 实验名
def do_experiment(init_num_experts):

    experiment_name = "D_{}_random".format(init_num_experts) # get_name(init_num_experts,d_out,need_pretrain)

    save_dir = save_dir_general + "_" + experiment_name

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, "params_and_config"+".pth")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # 评估
    progress_bar = tqdm(dataloader)
    with torch.no_grad():
        label_cluster = []
        for x_batch, label in progress_bar:
            x_batch = x_batch.to(device)
            B = x_batch.shape[0]
            merged_expert = torch.randint(0,init_num_experts,(B,))
            label_cluster += [(label[b], merged_expert[b]) for b in range(x_batch.shape[0])]

    name_heatmap = "heatmap_" + experiment_name + ".pdf"
    save_path_heatmap = os.path.join(save_dir, name_heatmap)

    nmi, ari, purity = get_nmi_ari_purity(label_cluster)
    cm = save_confusion_matrix_heatmap(label_cluster, save_path_heatmap)
    # 保存txt
    text_name = experiment_name + ".txt"
    text_path = os.path.join(save_dir, text_name)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("=== 聚类评价指标 ===\n")
        f.write(f"NMI: {nmi:.4f}\n")
        f.write(f"ARI: {ari:.4f}\n")
        f.write(f"Purity: {purity:.4f}\n\n")
        f.write("=== 混淆矩阵 ===\n")
        for row in cm:
            f.write("\t".join(map(str, row)) + "\n")
    print("实验完成")



do_experiment(5)





'''
monopoly.eval()
with torch.no_grad():
    label_cluster = []
    for x_batch, label in dataloader:
        x_batch = x_batch.to(device)
        loss, expert_select = monopoly.get_loss(x_batch, kl_weight=weight_kl)
        # expert_select: (B,J) int: index of expert (joint wise), J:joint index
        label_cluster += [{"label":label[b], "cluster":expert_select[b]} for b in range(x_batch.shape[0])]


with open("label_cluster.txt", "w") as f:
    for item in label_cluster:
        label = item["label"]
        cluster = item["cluster"].tolist()  # 转为 list
        f.write(f"label: {label.tolist()}, cluster: {cluster}\n")

num_visual = 10
save_dir = "./experiment_save"
# 每个样本创建一个文件夹 bvh & cluster_result
# 保存一个bvh和一个txt文件

for epoch in range(num_visual):
    filename = str(epoch)
    x, label = dataset.get_a_rand_data()
    x = x.to(device) # (T,N,d)
    x_9d = dataset.processor.trans_6d_to_9d(x)
'''

'''
self = monopoly

B,dT,N,d = x_batch.shape
assert dT==self.time_len_merge
total_loss = 0
joint_expert_idx = []
for j in range(self.joint_num):
    loss, expert_idx = self.get_j_loss(x_batch,j,kl_weight)
    total_loss += loss
    joint_expert_idx += [expert_idx]
total_loss = total_loss / self.joint_num
# (J,B) -> (B,J)
joint_expert_idx = torch.stack(joint_expert_idx)
joint_expert_idx = joint_expert_idx.to(torch.long)
'''