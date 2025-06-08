import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from MLP import MLP
from sklearn.decomposition import PCA

from dummy_dataset import FactorizedDataset
from vae import VariationalAutoEncoder, vae_loss, auxiliary_projection_loss

import os

def get_name(mlp_dataset, mlp_model, weight_kl):
    return "datasetMLP_{}_modelMLP_{}_KLweight_{}".format(mlp_dataset, mlp_model, weight_kl)




save_dir = "./experiment1_save"
input_dim=54
latent_dim=8
batch_size=128
epochs=200

def do_experiment(num_mlp_layers_dataset, num_mlp_layers_model, weight_kl):
    experiment_name = get_name(num_mlp_layers_dataset, num_mlp_layers_model, weight_kl)

    dataset = FactorizedDataset(n_samples=5000, z_dim=latent_dim, x_dim=input_dim, num_mlp_layers=num_mlp_layers_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = VariationalAutoEncoder(input_dim, latent_dim, num_mlp_layers_model)

    device = torch.device("cuda")
    vae.to(device)

    all_losses = []

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            x_hat, mu, logvar = vae(x_batch)
            # 主训练
            loss, recon, kl = vae_loss(x_hat, x_batch, mu, logvar, weight_kl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 辅助训练
            mu = mu.detach()
            mu_recons = vae.recons_z(mu)
            loss_vis = auxiliary_projection_loss(mu, mu_recons)
            optimizer.zero_grad()
            loss_vis.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss={loss.item():.4f}")


    vae.to('cpu')
    x, z_true = dataset[:]
    with torch.no_grad():
        x_hat, mu, logvar = vae(x)
        z_pred = mu

    # 1. 原始和重构点云（reshape为25×2）
    x0 = x[0].reshape(input_dim//2, 2)
    x0_hat = x_hat[0].reshape(input_dim//2, 2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x0[:, 0], x0[:, 1], label="Original")
    plt.title("Original Point Cloud")
    plt.subplot(1, 2, 2)
    plt.scatter(x0_hat[:, 0], x0_hat[:, 1], label="Reconstructed", color="orange")
    plt.title("Reconstructed Point Cloud reconsLoss={:.4f}".format(loss.item()))
    plt.tight_layout()
    save_name = "recons_" + experiment_name + ".pdf"
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



num_mlp_layers_dataset = 3
num_mlp_layers_model = 3
kl_weights = 0.01


# do_experiment(num_mlp_layers_dataset=3, num_mlp_layers_model=3, weight_kl=0)
# do_experiment(num_mlp_layers_dataset=3, num_mlp_layers_model=3, weight_kl=kl_weights)
# 
# do_experiment(num_mlp_layers_dataset=3, num_mlp_layers_model=2, weight_kl=0)
# do_experiment(num_mlp_layers_dataset=3, num_mlp_layers_model=2, weight_kl=kl_weights)
# 
# do_experiment(num_mlp_layers_dataset=2, num_mlp_layers_model=3, weight_kl=0)
# do_experiment(num_mlp_layers_dataset=2, num_mlp_layers_model=3, weight_kl=kl_weights)

do_experiment(num_mlp_layers_dataset=2, num_mlp_layers_model=2, weight_kl=0)
# do_experiment(num_mlp_layers_dataset=2, num_mlp_layers_model=2, weight_kl=kl_weights)

