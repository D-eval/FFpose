import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from MLP import MLP
from sklearn.decomposition import PCA

from dummy_hierarchy_dataset import HierarchyDataset
from monopolyMoE_Deep import DeepMonoMoE

from eval_cluster import get_nmi_ari_purity, save_confusion_matrix_heatmap

import os

def get_name(layers_cls, layers_experts, weight_kl, need_pretrain, layers_dim_data, layers_dim_model):
    return "trueCls_{}_expertCls_{}_dimData_{}_dimModel_{}_KLweight_{}_needPretrain_{}".format(str(layers_cls),
                                                                                               str(layers_experts),
                                                                                               str(layers_dim_data),
                                                                                               str(layers_dim_model),
                                                                                               weight_kl,
                                                                                               need_pretrain)

# 计算 

save_dir = "./experiment5_save"

num_mlp_layers = 3

batch_size=128
epochs=200

num_pretrain = 100

n_each_cls=1000
num_mlp_layers=3


layers_dim=[8,16,54]
layers_dim_model=[54,16,8]


def do_experiment(layers_cls, layers_experts,
                  layers_dim_data, layers_dim_model,
                  weight_kl, need_pretrain,):
    experiment_name = get_name(layers_cls, layers_experts, weight_kl, need_pretrain,
                               layers_dim_data, layers_dim_model)

    dataset = HierarchyDataset(n_each_cls = n_each_cls,
                               num_mlp_layers = num_mlp_layers,
                               layers_cls = layers_cls,
                               layers_dim = layers_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeepMonoMoE(layers_experts, layers_dim_model, num_mlp_layers)

    device = torch.device("cuda")
    model.to(device)



    # 训练各层:
    for l in range(model.num_layers):
        # 各自贴到某个样本上
        if need_pretrain:
            for e, expert in enumerate(model.all_layers[l].all_experts):
                optimizer = torch.optim.Adam(expert.parameters(), lr=1e-3)
                x, label = dataset.get_a_rand_data()
                x = x.to(device).unsqueeze(0)
                with torch.no_grad():
                    x, _, _ = model(x, l)
                x = x.detach()
                for epoch in range(num_pretrain):
                    loss, _ = expert.get_recon_loss(x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print(e,loss.item())
            print("预训练结束")
        else:
            print("跳过预训练")
        print("开始第{}层重构训练".format(l))
        all_losses = []
        optimizer = torch.optim.Adam(model.all_layers[l].parameters(), lr=1e-3)
        for epoch in range(epochs):
            for x_batch, label in dataloader:
                x_batch = x_batch.to(device)
                loss, expert_select = model.get_loss(x_batch, deep=l, kl_weight=weight_kl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(expert_select)
                print(f"Epoch {epoch}, loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        label_cluster = []
        for x_batch, label in dataloader:
            x_batch = x_batch.to(device)
            loss, expert_select = model.get_loss(x_batch, deep=model.num_layers-1, kl_weight=weight_kl)
            label_cluster += [(label[b,0], expert_select[b]) for b in range(x_batch.shape[0])]

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




# 有kl散度时
do_experiment([2,3],[3,2],[8,16,54],[54,16,8],0.01,True)

# 刚好相等时
do_experiment([2,3],[3,2],[8,16,54],[54,16,8],0,True)

# 增强模型力
do_experiment([2,3],[3,2],[8,16,54],[54,32,8],0,True)

# 不知多少类
do_experiment([2,3],[4,4],[8,16,54],[54,16,8],0,True)

'''
layers_cls = [2,3]
layers_experts = [3,2]
layers_dim_data = [8,16,54]
layers_dim_model = [54,16,8]
weight_kl = 0.01
need_pretrain = True

experiment_name = get_name(layers_cls, layers_experts, weight_kl, need_pretrain,
                            layers_dim_data, layers_dim_model)

dataset = HierarchyDataset(n_each_cls = n_each_cls,
                            num_mlp_layers = num_mlp_layers,
                            layers_cls = layers_cls,
                            layers_dim = layers_dim)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DeepMonoMoE(layers_experts, layers_dim_model, num_mlp_layers)

device = torch.device("cuda")
model.to(device)



# 训练各层:
for l in range(model.num_layers):
    # 各自贴到某个样本上
    if need_pretrain:
        for e, expert in enumerate(model.all_layers[l].all_experts):
            optimizer = torch.optim.Adam(expert.parameters(), lr=1e-3)
            x, label = dataset.get_a_rand_data()
            x = x.to(device).unsqueeze(0)
            with torch.no_grad():
                x, _, _ = model(x, l)
            x = x.detach()
            for epoch in range(num_pretrain):
                loss, _ = expert.get_recon_loss(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(e,loss.item())
        print("预训练结束")
    else:
        print("跳过预训练")
    print("开始第{}层重构训练".format(l))
    all_losses = []
    optimizer = torch.optim.Adam(model.all_layers[l].parameters(), lr=1e-3)
    for epoch in range(epochs):
        for x_batch, label in dataloader:
            x_batch = x_batch.to(device)
            loss, expert_select = model.get_loss(x_batch, deep=l, kl_weight=weight_kl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(expert_select)
            print(f"Epoch {epoch}, loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    label_cluster = []
    for x_batch, label in dataloader:
        x_batch = x_batch.to(device)
        loss, expert_select = model.get_loss(x_batch, deep=model.num_layers-1, kl_weight=weight_kl)
        label_cluster += [(label[b,0], expert_select[b]) for b in range(x_batch.shape[0])]


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

'''