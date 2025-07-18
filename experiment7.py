# watch -n 1 nvidia-smi

import torch

from read_data import AMASSDataset

from models.STmonopolyMoE_Deep import DeepME
from configs.group_joint_dict import group_joint_dict
from configs.group2_group_dict import group2_group_dict

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

save_dir_general = "../experiment_save/experiment7"

num_pretrain = 500
epochs = 1
lr = 1e-3

threshold_loss = 0.05
init_num_experts = 5

d_mid = 8

d_final = 16

weight_kl = 0
need_train = True # True
num_visual = 10 # 可视化的个数 （写bvh文件）




layer1_save_path = "/home/vipuser/DL/Dataset100G/DL-3D-Upload/model/point-cloud-motion/experiment_save/experiment6_E_5_D_8_pretrain_True/params_and_config.pth"

# 实验参数
d_out = 32
need_pretrain = True # True
num_experts_second = 50


dataset = AMASSDataset(data_path,
                         device=device,
                         time_len=1.5,
                         use_hand=False,
                         target_fps=2,
                         use_6d=True)

idx_to_show = torch.randint(0,len(dataset)-1,(num_visual,))

# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_name(epoch,num_experts_second,d_out,need_pretrain):
    experiment_name = "epoch_{}_E_{}_D_{}_pretrain_{}".format(epoch,num_experts_second,d_out,need_pretrain)
    return experiment_name


# 实验名
def do_experiment(epoch,num_experts_second,d_out,need_pretrain):
    group_num_experts = {g:init_num_experts for g in group_joint_dict.keys()}

    experiment_name = get_name(epoch,num_experts_second,d_out,need_pretrain)

    save_dir = save_dir_general + "_" + experiment_name

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, "params_and_config"+".pth")

    # 加载状态
    if os.path.exists(save_path):
        save_dict = torch.load(save_path)
        monopoly = DeepME(**save_dict["config"]) # *拆列表元组，**拆字典
        monopoly.load_state_dict(save_dict["params"])
        print("成功加载参数")
    else:
        save_dict = {
            "params":None,
            "config":{
                "group_joint_dict":group_joint_dict,
                "group2_group_dict":group2_group_dict,
                "group_num_experts":group_num_experts,
                "d_in":d_in, "d_mid":d_mid,
                "time_len_merge":time_len_merge,
                "mlp_layers":mlp_layers,
                "num_experts_second":num_experts_second,
                "d_final":d_final,
            }
        }
        monopoly = DeepME(**save_dict["config"])
        layer1_save_dict = torch.load(layer1_save_path)
        layer1_params = layer1_save_dict["params"]
        monopoly.load_layers(layer1_params)
        print("没有加载参数")

    monopoly.to(device)

    # 各自贴到某个样本上
    # 有妈的才分化，没妈的都一样
    # 让AE都有妈
    if need_pretrain:
        for e in range(num_experts_second):
            params = monopoly.get_train_params()
            optimizer = torch.optim.Adam(params, lr=1e-4) # pretrain lr 略低一些
            x, label = dataset.get_a_rand_data()
            x = x.to(device).unsqueeze(0)
            for epoch in range(num_pretrain):
                loss = monopoly.soldier_step_out(x,e)
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
                print(expert_select[:100])
                assert expert_select.to(torch.float).std() != 0, "模式坍缩"
            print(f"Epoch {epoch}, loss={loss.item():.4f}")
            save_dict["params"] = monopoly.state_dict()
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
            _, z2, _, _, e1, e2, _ = monopoly(x.unsqueeze(0))
            xhat, _ = monopoly.decode(z2, e2, e1)
        save_bvh(x,file_dir,"real.bvh")
        save_bvh(xhat[0],file_dir,"recons.bvh")
        with open(os.path.join(file_dir,"label_cluster.txt"), "w") as f:
            f.write("label: {}\n".format(label))
            f.write("cluster: {}\n".format(e2.item()))
            f.write("joint expert:\n")
            for g,e in e1.items():
                f.write("{}:{}\n".format(g, e.item()))

    # 评估
    monopoly.eval()
    with torch.no_grad():
        label_cluster = []
        for x_batch, label in dataloader:
            x_batch = x_batch.to(device)
            loss, expert_select = monopoly.get_loss(x_batch, kl_weight=weight_kl)
            label_cluster += [(label[b], expert_select[b]) for b in range(x_batch.shape[0])]


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



# 实验参数
# d_out = 32
# need_pretrain = True # True
# num_experts_second = 50

# do_experiment(50, 32, True)

do_experiment(1, 5, 8, True)
do_experiment(2, 5, 8, True)


'''
do_experiment(10, 64, True)
do_experiment(20, 64, True)
do_experiment(10, 64, False)
'''

'''
self = monopoly
B,T,N,d = x.shape
assert B==1
z1_dict, _, _, g1_e = self.layer1(x)
z1 = self.trans_dict_to_Tensor(z1_dict) # (B,G,D)
z1 = torch.flatten(z1,1,2)

z = z1_dict

all_z = []
all_group = self.group2_group_dict["all"]
for g in all_group:
    all_z += z[g]


# [(B,D) x G]
z = torch.stack(all_z,dim=1) # (B,G,D)
'''
