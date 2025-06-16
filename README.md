# 记忆融合机制 (Memory Fusion Mechanism, MFM)

本项目用于本科毕设。

## 算法思想

通过多个AE重构数据，损失小的更新。
每个AE就是一个簇，
当出现新数据时，计算重构损失就是衡量距离。

## 数据集

### 真实数据

在AMASS的CMU数据集上训练，到AMASS的官网下载。

### 模拟数据

在models.dummy_dataset中实现了虚拟数据集
用于测试本方法。