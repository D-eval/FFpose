from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import Counter, defaultdict
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def compute_nmi(label_cluster):
    true_labels, cluster_labels = zip(*label_cluster)
    return normalized_mutual_info_score(true_labels, cluster_labels)

def compute_ari(label_cluster):
    true_labels, cluster_labels = zip(*label_cluster)
    return adjusted_rand_score(true_labels, cluster_labels)

def compute_purity(label_cluster):
    true_labels, cluster_labels = zip(*label_cluster)
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    cluster_to_indices = defaultdict(list)
    for idx, c in enumerate(cluster_labels):
        cluster_to_indices[c].append(idx)

    correct = 0
    for indices in cluster_to_indices.values():
        cluster_true_labels = true_labels[indices]
        majority_label = Counter(cluster_true_labels).most_common(1)[0][0]
        correct += np.sum(cluster_true_labels == majority_label)

    return correct / len(label_cluster)


def save_confusion_matrix_heatmap(label_cluster, save_path):
    true_labels, cluster_labels = zip(*label_cluster)
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    # 获取混淆矩阵
    cm = confusion_matrix(true_labels, cluster_labels)

    # 可视化热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(cluster_labels),
                yticklabels=np.unique(true_labels))

    plt.xlabel("Expert Code")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of Expert Code vs True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return cm

def get_nmi_ari_purity(label_cluster):
    nmi = compute_nmi(label_cluster)
    ari = compute_ari(label_cluster)
    purity = compute_purity(label_cluster)
    return nmi, ari, purity

