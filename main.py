import torch
import utils
import time
import numpy as np
import torch.nn.functional as F
import torch_geometric.utils as gutils
from model import NLGNN


if __name__ == '__main__':
    # dataset_name = 'chameleon'
    dataset_name = 'actor'

    heter_dataset = ['chameleon', 'cornell', 'actor', 'squirrel', 'texas', 'wisconsin']  # 异配图
    homo_dataset = ['Cora', 'Citeseer', 'Pubmed']  # 同配图
    # 异配图(disassortative graphs)是指相似的节点之间有更多的连接，而不相似的节点之间连接较少。这种类型的图被称为异配图(disassortative graphs) 。
    # 相反，同配图(assortative graphs)是指相似的节点之间有较少的连接，而不相似的节点之间连接较多。这种类型的图被称为同配图(assortative graphs) 。

    lr = 0.01
    weight_decay = 5e-4
    max_epoch = 500
    patience = 200  # 如果超过这么多轮次没有提升，直接停
    num_hidden = [92, 48, 16, 8]
    dropout = 0
    le_list = ['mlp', 'gcn', 'gat']
    # 多层感知机(MLP)、图卷积网络(GCN)、自注意力网络(GAT)
    le = le_list[0]
    print(f"此时模型是{le}")
    # 目前，window_size大小应该是奇数，否则我们需要修改Conv1d中的“padding”设置。
    window_size = 5

    re_generate_train_val_test = True

    split_by_label_flag = True
    if dataset_name in ['chameleon', 'cornell', 'texas']:
        split_by_label_flag = False

    if dataset_name in heter_dataset:
        data, num_features, num_classes = utils.load_heter_data(dataset_name)
        print("此时进行的是{} dataset.".format(dataset_name))
    elif dataset_name in homo_dataset:
        dataset = utils.load_homo_data(dataset_name)
        data = dataset[0]  # 形式：Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        print("此时进行的是{} dataset.".format(dataset_name))
    else:
        print("我们现在没有{} dataset.".format(dataset_name))

    utils.set_seed(15)

    print(data)
    print(f'特征{num_features}，分类数{num_classes}')

    if re_generate_train_val_test:
        idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.55, 0.25, 0.2, 15, split_by_label_flag)
        # 这里自定义的一个函数，0.6代表的是概率
    else:
        if dataset_name in heter_dataset:
            idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.55, 0.25, 0.2, 15, split_by_label_flag)
            # idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.48, 0.32, 0.20, 15, split_by_label_flag)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data = data.to(device)
    # x = torch.from_numpy(x).to(device)
    idx_train = torch.from_numpy(idx_train)
    idx_val = torch.from_numpy(idx_val)
    idx_test = torch.from_numpy(idx_test)

    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    data.edge_index = gutils.remove_self_loops(data.edge_index)[0]
    # 函数返回的第一个元素是删除自环后的边索引，第二个元素是包含自环的索引
    net = NLGNN(data, le, window_size, num_features, num_hidden, num_classes, dropout)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # 权重衰减(Weight Decay)是一种正则化技术，用于防止神经网络过拟合。
    # 它具体来说，权重衰减会使得模型的参数更新变得更加缓慢，从而减少了模型对训练数据的过度拟合。
    # 系数越大，权重衰减的效果就越明显

    dur = []
    los = []
    loc = []
    counter = 0
    min_loss = 100.0
    max_acc = 0.0

    for epoch in range(max_epoch):
        if epoch >= 3:
            t0 = time.time()
        net.train()
        logp = net()

        cla_loss = F.nll_loss(logp[idx_train], data.y[idx_train])
        loss = cla_loss
        train_acc = utils.accuracy(logp[idx_train], data.y[idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        logp = net()
        test_acc = utils.accuracy(logp[idx_test], data.y[idx_test])
        loss_val = F.nll_loss(logp[idx_val], data.y[idx_val]).item()
        val_acc = utils.accuracy(logp[idx_val], data.y[idx_val])
        los.append([epoch, loss_val, val_acc, test_acc])

        if loss_val < min_loss and max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience and dataset_name in homo_dataset:  # 同配图
            print('同配图，由于很长时间模型没更新，停！')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        if (epoch+1) % 100 == 0:
            print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
                epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))

    if dataset_name in homo_dataset or 'syn' in dataset_name:
        los.sort(key=lambda x: x[1])  # los.append([epoch, loss_val, val_acc, test_acc])
        print(los)
        acc = los[0][-1]  # 损失最少的那一个的测试准确率
        print(f"最好模型测试准确率{acc}")
    else:
        los.sort(key=lambda x: -x[2])  # 负号代表要降序
        print(los)
        acc = los[0][-1]
        print(f"最好模型测试准确率{acc}")

    print(f"此时模型是{le}")
    print("此时数据集是{} dataset.".format(dataset_name))