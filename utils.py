import torch
import numpy as np
import torch_geometric
import random

from scipy import io
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split

from torch.backends import cudnn


def load_heter_data(dataset_name):
    DATAPATH = 'data/heterophily_datasets_matlab'
    fulldata = io.loadmat(f'{DATAPATH}/{dataset_name}.mat')

    edge_index = fulldata['edge_index']  # [[],[]]
    node_feat = fulldata['node_feat']  # （节点，每个节点特征向量）
    label = np.array(fulldata['label'], dtype=np.int32).flatten()  # flatten将向量铺平，铺成一维度的。

    num_features = node_feat.shape[1]  # 节点的特征数
    num_classes = np.max(label) + 1  # 分类数
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # 转换成tensor
    x = torch.tensor(node_feat)
    y = torch.tensor(label, dtype=torch.long)

    edge_index = torch_geometric.utils.to_undirected(edge_index)  # 将有向图转换为无向图
    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)  # 用于从一个图中移除所有的自环
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)  # 将数据封装好了。
    return data, num_features, num_classes


def load_homo_data(dataset_name):
    dataset = Planetoid(root='./tmp/'+dataset_name, name=dataset_name)
    return dataset


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    # CUDNN的自动调优功能禁用。这将使CUDNN始终使用固定的卷积算法和数据类型，而不是根据输入数据动态调整。请注意，禁用自动调优功能可能会降低计算效率。
    cudnn.deterministic = True
    # 用于控制是否启用CUDNN的确定性模式。在确定性模式下，CUDNN将始终使用相同的卷积算法和数据类型来处理输入数据，以确保每次运行程序时生成的输出结果相同。这对于调试和复现实验结果非常有用
    return seed


def split_nodes(labels, train_ratio, val_ratio, test_ratio, random_state, split_by_label_flag):
    # (data.y,   0.6,          0.2,      0.2,         15,        split_by_label_flag)
    idx = torch.arange(labels.shape[0])
    idx = idx.numpy()
    if split_by_label_flag:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio, stratify=labels)
        # 根据labels数组中的类标签进行分层，以确保训练和测试数据集中每个类别的比例与原始数据集中每个类别的比例相同。
        # random_state参数用于控制在应用拆分之前对数据进行洗牌的随机性。 随机种子
    else:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio)

    if val_ratio:
        labels_train_val = labels[idx_train]
        if split_by_label_flag:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio), stratify=labels_train_val)
        else:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio))
    else:
        idx_val = None

    return idx_train, idx_val, idx_test


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    # correct.item()将张量转换为标量，防止数据类型不对报错
    return correct.item()*1.0/len(labels)
