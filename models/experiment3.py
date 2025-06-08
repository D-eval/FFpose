import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from MLP import MLP
from sklearn.decomposition import PCA

from dummy_class_dataset import DummyClassDataset
from monopolyMoE import MonoMoE

import os

def get_name(cls_num, num_experts, weight_kl, need_pretrain):
    return "clsNum_{}_expertNum_{}_KLweight_{}_needPretrain_{}".format(cls_num, num_experts, weight_kl, need_pretrain)



save_dir = "./experiment3_save"
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
                print(e,loss.item())
        print("预训练结束")
    else:
        print("跳过预训练")

    all_losses = []
    optimizer = torch.optim.Adam(monopoly.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            loss, expert_select = monopoly.get_loss(x_batch, kl_weight=weight_kl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(expert_select)
            print(f"Epoch {epoch}, loss={loss.item():.4f}")


    monopoly.to('cpu')
    x, label = dataset[:]
    with torch.no_grad():
        mu, logvar, x_hat = monopoly(x)
        z_pred = mu
        
    z_true = dataset.z

    # 1. 原始和重构点云（reshape为25×2）
    for c in range(cls_num):
        idx, _ = dataset.get_cls_idx(c)
        x0 = x[idx].reshape(input_dim//2, 2)
        x0_hat = x_hat[idx].reshape(input_dim//2, 2)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(x0[:, 0], x0[:, 1], label="Original")
        plt.title("Original Point Cloud")
        plt.subplot(1, 2, 2)
        plt.scatter(x0_hat[:, 0], x0_hat[:, 1], label="Reconstructed", color="orange")
        plt.title("Reconstructed Point Cloud reconsLoss={:.4f}".format(loss.item()))
        plt.tight_layout()
        save_name = "recons_" + experiment_name + '_class_{}'.format(c) + ".pdf"
        plt.savefig(os.path.join(save_dir, save_name))
        plt.close()

    # 2. z_true 和 mu 的 PCA 降维对比
    pca = PCA(n_components=2)
    z_true_2d = pca.fit_transform(z_true.numpy())
    z_pred_2d = pca.transform(z_pred.numpy())
    plt.figure(figsize=(6, 6))
    plt.scatter(z_true_2d[:, 0], z_true_2d[:, 1], label="True z", alpha=0.5)
    plt.scatter(z_pred_2d[:, 0], z_pred_2d[:, 1], label="Pred z (mu)", alpha=0.5)
    plt.legend()
    plt.title("True vs Predicted Latent (PCA)")
    plt.grid(True)
    save_name = "vae_" + experiment_name + ".pdf"
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()


kl_weights = 0.01

# do_experiment(cls_num=2, num_experts=2, weight_kl=0, need_pretrain=True)
# do_experiment(cls_num=2, num_experts=2, weight_kl=0.01, need_pretrain=True)
# do_experiment(cls_num=2, num_experts=2, weight_kl=0, need_pretrain=False)
# do_experiment(cls_num=2, num_experts=1, weight_kl=0, need_pretrain=False)
# do_experiment(cls_num=1, num_experts=2, weight_kl=0, need_pretrain=True)

# do_experiment(cls_num=1, num_experts=1, weight_kl=0, need_pretrain=True)

do_experiment(cls_num=1, num_experts=2, weight_kl=0, need_pretrain=False)
