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


init_num_experts = 5
group_num_experts = {g:init_num_experts for g in group_joint_dict.keys()}
d_in = 6

time_len_merge = 3
mlp_layers = 3

batch_size = 128

kl_weight = 0.01

device = torch.device("cuda")

data_path = ["/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/CMU"]

save_dir = "../experiment_save/experiment6"


num_pretrain = 100
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

# 实验名
def do_experiment(d_out,need_pretrain):
    experiment_name = "D_{}_pretrain_{}".format(d_out,need_pretrain)

    save_dir += "_" + experiment_name

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, "params_and_config"+".pth")


    # 加载状态
    if os.path.exists(save_path):
        save_dict = torch.load(save_path)
        monopoly = GlobalmonopolyMoE(**save_dict["config"]) # *拆列表元组，**拆字典
        monopoly.load_state_dict(save_dict["params"])
        print("成功加载参数")
    else:
        save_dict = {
            "params":None,
            "config":{
                "group_joint_dict":group_joint_dict,
                "group_num_experts":group_num_experts,
                "d_in":d_in,
                "d_out":d_out,
                "time_len_merge":time_len_merge,
                "mlp_layers":mlp_layers
            }
        }
        monopoly = GlobalmonopolyMoE(**save_dict["config"])
        print("没有加载参数")


    monopoly.to(device)


    # 各自贴到某个样本上
    # 有妈的才分化，没妈的都一样
    # 让AE都有妈
    if need_pretrain:
        for e in range(init_num_experts):
            params = monopoly.get_e_parameters(e)
            optimizer = torch.optim.Adam(params, lr=1e-4) # pretrain lr 略低一些
            x, label = dataset.get_a_rand_data()
            x = x.to(device).unsqueeze(0)
            for epoch in range(num_pretrain):
                loss = monopoly.team_step_out(x,e)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch%10==0:
                    print(e,epoch,loss.item())
        print("预训练结束")
    else:
        print("跳过预训练")


    # 训练
    if need_train:
        # all_losses = []
        optimizer = torch.optim.Adam(monopoly.parameters(), lr=lr)
        for epoch in range(epochs):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            progress_bar = tqdm(dataloader, desc="Training", leave=False)
            for x_batch, label in progress_bar:
                x_batch = x_batch.to(device)
                loss, expert_select = monopoly.get_loss(x_batch, kl_weight=weight_kl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                assert expert_select['leg_L'].to(torch.float).std() != 0, "模式坍缩"
            print(f"Epoch {epoch}, loss={loss.item():.4f}")
            save_dict["params"] = monopoly.state_dict()
            save_dict["config"]["group_num_experts"] = monopoly.group_num_experts
            torch.save(save_dict, save_path)
    else:
        print("跳过训练")


    monopoly.eval()
    for epoch in range(num_visual):
        idx = idx_to_show[epoch]
        x, label = dataset[idx]
        x = x.to(device) # (T,N,d)
        filename = str(epoch)
        file_dir = os.path.join(save_dir,filename)
        with torch.no_grad():
            z,_,xhat,expert_idx = monopoly(x.unsqueeze(0))
        save_bvh(x,file_dir,"real.bvh")
        save_bvh(xhat[0],file_dir,"recons.bvh")
        with open(os.path.join(file_dir,"label_cluster.txt"), "w") as f:
            f.write("label: {}\n".format(label))
            f.write("joint expert:\n")
            for g,e in expert_idx.items():
                f.write("{}:{}\n".format(g, e.item()))
    print("实验完成")



do_experiment(8,True)
do_experiment(32,True)
do_experiment(32,False)


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