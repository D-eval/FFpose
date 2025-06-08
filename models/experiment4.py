import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from MLP import MLP
from sklearn.decomposition import PCA

from dummy_class_dataset import DummyClassDataset
from monopolyMoE import MonoMoE

from eval_cluster import get_nmi_ari_purity, save_confusion_matrix_heatmap

import os

def get_name(cls_num, num_experts, weight_kl, need_pretrain):
    return "clsNum_{}_expertNum_{}_KLweight_{}_needPretrain_{}".format(cls_num, num_experts, weight_kl, need_pretrain)

# 计算 

save_dir = "./experiment4_save"
input_dim=54
latent_model = 8
num_mlp_layers = 3
n_samples = 10000
batch_size=128
epochs=200

num_pretrain = 100


def do_experiment(cls_num, num_experts, weight_kl, need_pretrain):
    experiment_name = get_name(cls_num, num_experts, weight_kl, need_pretrain)

    dataset = DummyClassDataset(n_samples=n_samples, z_dim=latent_model, x_dim=input_dim, num_mlp_layers=num_mlp_layers,
                                cls_num=cls_num)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    monopoly = MonoMoE(num_experts, input_dim, latent_model, num_mlp_layers)

    device = torch.device("cuda")
    monopoly.to(device)

    # 各自贴到某个样本上
    if need_pretrain:
        for e, expert in enumerate(monopoly.all_experts):
            optimizer = torch.optim.Adam(expert.parameters(), lr=1e-3)
            x, label = dataset.get_a_rand_data()
            x = x.to(device).unsqueeze(0)
            for epoch in range(num_pretrain):
                loss, _ = expert.get_recon_loss(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(e,loss.item())
        print("预训练结束")
    else:
        print("跳过预训练")

    all_losses = []
    optimizer = torch.optim.Adam(monopoly.parameters(), lr=1e-3)
    for epoch in range(epochs):
        # label_cluster = []
        for x_batch, label in dataloader:
            x_batch = x_batch.to(device)
            loss, expert_select = monopoly.get_loss(x_batch, kl_weight=weight_kl)
            # label_cluster += [(label[b], expert_select[b]) for b in range(x_batch.shape[0])]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(expert_select)
            print(f"Epoch {epoch}, loss={loss.item():.4f}")

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


kl_weights = 0.01

do_experiment(cls_num=2, num_experts=2, weight_kl=0.01, need_pretrain=False)
do_experiment(cls_num=3, num_experts=3, weight_kl=0.01, need_pretrain=False)

do_experiment(cls_num=2, num_experts=2, weight_kl=0, need_pretrain=True)
do_experiment(cls_num=3, num_experts=3, weight_kl=0, need_pretrain=True)