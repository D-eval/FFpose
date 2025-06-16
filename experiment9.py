
import torch

from read_data import AMASSDataset


from models.STmonopolyMoE_1 import GlobalmonopolyMoE
from configs.center_neighbor_idx_dict import center_neighbor_idx_dict_smpl as joint_neighbor_dict

from torch.utils.data import DataLoader, TensorDataset

from models.eval_cluster import get_nmi_ari_purity, save_confusion_matrix_heatmap

import os


def save_bvh(x,file_dir,save_name):
    # x: (T,N,d)
    x_9d = dataset.processor.trans_6d_to_9d(x)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    dataset.processor.write_bvh(x_9d,
                                filename=save_name,
                                save_dir=file_dir)



num_experts = 5
d_in = 6
d_out = 64
time_len_merge = 3
mlp_layers = 3

batch_size = 128

kl_weight = 0.01

device = torch.device("cuda")

data_path = ["/home/vipuser/DL/Dataset100G/AMASS/SMPL_H_G/CMU"]

save_dir = "../experiment_save/experiment9"


num_pretrain = 300
epochs = 12
lr = 1e-3


weight_kl = 0.01
need_pretrain = True
need_train = True

num_visual = 10 # 可视化的个数 （写bvh文件）


dataset = AMASSDataset(data_path,
                         device=device,
                         time_len=1.5,
                         use_hand=False,
                         target_fps=2,
                         use_6d=True)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


monopoly = GlobalmonopolyMoE(joint_neighbor_dict=joint_neighbor_dict,
                             num_experts=num_experts,
                             d_in=d_in,d_out=d_out,
                             time_len_merge=time_len_merge,mlp_layers=mlp_layers)

monopoly.to(device)


device = torch.device("cuda")
monopoly.to(device)

# 各自贴到某个样本上
# 有妈的才分化，没妈的都一样
# 让AE都有妈
if need_pretrain:
    for e in range(monopoly.num_experts):
        params = monopoly.get_e_parameters(e)
        optimizer = torch.optim.Adam(params, lr=1e-4) # pretrain lr 略低一些
        x, label = dataset.get_a_rand_data()
        x = x.to(device).unsqueeze(0)
        for epoch in range(num_pretrain):
            loss = monopoly.team_step_out(x,e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(e,loss.item())
    print("预训练结束")
else:
    print("跳过预训练")


# 实验名
experiment_name = "E_{}_T_{}_D_{}".format(num_experts,time_len_merge,d_out)
save_path = os.path.join(save_dir, experiment_name+".pth")

# 加载状态
if os.path.exists(save_path):
    save_dict = torch.load(save_path)
    monopoly.load_state_dict(save_dict["params"])
else:
    save_dict = {
        "params":monopoly.state_dict(),
        "config":{
            "joint_neighbor_dict":joint_neighbor_dict,
            "num_experts":num_experts,
            "d_in":d_in,
            "d_out":d_out,
            "time_len_merge":time_len_merge,
            "mlp_layers":mlp_layers
        }
    }

# 训练
if need_train:
    all_losses = []
    optimizer = torch.optim.Adam(monopoly.parameters(), lr=lr)
    for epoch in range(epochs):
        for x_batch, label in dataloader:
            x_batch = x_batch.to(device)
            loss, expert_select = monopoly.get_loss(x_batch, kl_weight=weight_kl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(expert_select)
        print(f"Epoch {epoch}, loss={loss.item():.4f}")
        save_dict["params"] = monopoly.state_dict()
        torch.save(save_dict, save_path)
else:
    print("跳过训练")



monopoly.eval()
for epoch in range(num_visual):
    x, label = dataset.get_a_rand_data()
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
        for j,e in enumerate(expert_idx[0]):
            f.write("{}:{}\n".format(j, e.item()))


def cal_cluster(cls_vector, joint_select=[2,12,19], E=5):
    # cls_vector: (B,J)
    # return: (B,)
    w = 1
    B = cls_vector.shape[0]
    cluster = torch.zeros((B,))
    for j in joint_select:
        cluster += cls_vector[:,j] * w
        w *= E
    return cluster
        

monopoly.eval()
with torch.no_grad():
    label_cluster = []
    for x_batch, label in dataloader:
        x_batch = x_batch.to(device)
        loss, expert_select = monopoly.get_loss(x_batch, kl_weight=weight_kl)
        # expert_select: (B,J) int: index of expert (joint wise), J:joint index
        cluster = cal_cluster(expert_select)
        label_cluster += [{"label":label[b], "cluster":cluster[b]} for b in range(x_batch.shape[0])]

label_cluster_1 = []
for dct in label_cluster:
    label_cluster_1 += [(dct["label"].item(), dct["cluster"].item())]

with open("label_cluster.txt", "w") as f:
    for item in label_cluster:
        label = item["label"]
        cluster = item["cluster"].tolist()  # 转为 list
        f.write(f"label: {label.tolist()}, cluster: {cluster}\n")


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