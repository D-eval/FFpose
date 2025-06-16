# 记忆融合机制 (Memory Fusion Mechanism, MFM)

本项目用于本科毕设。

## 算法思想

通过多个AE重构数据，损失小的更新。
每个AE就是一个簇，
当出现新数据时，计算重构损失就是衡量距离。

## 数据集

### 真实数据

在AMASS的CMU数据集上训练，到AMASS的官网下载。
还要下载SMPL模型，然后替换`read_data.py`中的`body_model_path`，

如果想要保存bvh文件，
还要更改`read_data.py`中的`write_bvh`和`get_pos_tn3`
函数里的`bdata`，
随便选一个AMASS数据集中的`npz`文件即可。

### 模拟数据

在models.dummy_dataset中实现了虚拟数据集
用于测试本方法。

## 代码解释

`experiment1`到`experiment5`在`./models`中

`experiment6`到`experimemt9`在`./`中

