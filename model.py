import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class NLGNN(torch.nn.Module):
    def __init__(self, data, le, window_size, num_features, num_hidden, num_classes, dropout):
        super(NLGNN, self).__init__()
        # num_hidden = [96, 48, 16, 8]
        if le == 'mlp':
            self.first_1 = torch.nn.Linear(num_features, num_hidden[0])  # num_features都是节点的特征数在变化
            self.first_2 = torch.nn.Linear(num_hidden[0], num_hidden[1])
        elif le == 'gcn':
            self.first_1 = GCNConv(num_features, num_hidden[0])
            self.first_2 = GCNConv(num_hidden[0], num_hidden[1])
        else: # 'gat'
            self.first_1 = GATConv(num_features, num_hidden[0])
            self.first_2 = GATConv(num_hidden[0], num_hidden[1])

        self.le = le
        self.attention_layer = torch.nn.Linear(num_hidden[1], 1)
        self.window_size = window_size
        self.conv1d1 = torch.nn.Conv1d(num_hidden[1], num_hidden[1], kernel_size=window_size, padding=int((self.window_size-1)/2))
        # 一维度的卷积层                # 通道数
        self.conv1d2 = torch.nn.Conv1d(num_hidden[1], num_hidden[1], kernel_size=window_size, padding=int((self.window_size-1)/2))

        self.final_layer = torch.nn.Linear(2 * num_hidden[1], num_classes)

        self.dropout = dropout
        self.data = data

    def forward(self):
        if self.le == 'mlp':
            h = self.first_1(self.data.x)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.first_2(h)
            # h = F.relu(h)
        else: #gcn or gat
            h = self.first_1(self.data.x, self.data.edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.first_2(h, self.data.edge_index)
            # h = F.relu(h)

        # h = F.dropout(h, p=self.dropout, training=self.training)

        before_h = h
        # self.attention_layer = torch.nn.Linear(num_hidden[1], 1)
        a = self.attention_layer(h)  # 把节点的特征数变成1 每个节点只有1个特征
        # h是一张图，它的shape通常是(batch_size, num_nodes, num_features)
        # 在此任务中shape通常是( num_nodes, num_features)
        # a的shape（batch_size, num_nodes, 1）
        sort_index = torch.argsort(a.flatten(), descending=True)
        # a.flatten()后 a的shape不会改变。 a.flatten()会将其变成1维，argsort将其排序 sort_index（atch_size*num_nodes，num_features）
        h = a * h
        # a的形状是(batch_size, num_nodes, 1),h的形状是(batch_size, num_nodes, num_features),它们的乘积结果的
        # 形状是(batch_size,num_nodes, num_features) 计算每个节点的注意力权重，即将每个节点的特征向量与对应的注意力权重相乘，得到加权特征向量

        h = h[sort_index].T.unsqueeze(0)
        # h 的形状将变为(0,num_features, num_nodes)  batch_size因为sort_index（atch_size*num_nodes，num_features）
        h = self.conv1d1(h)
        # conv1d1(h) 是对 h 的最后一维进行卷积操作。
        # 在这个例子中，h 经过转置和扩展维度后，它的形状变为 (1, num_hidden[1], num_nodes)。
        # 因此，conv1d1(h) 对 num_nodes 这一维进行卷积操作。 相当去去掉了一些无关的点

        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv1d2(h)  #torch.nn.Conv1d(num_hidden[1], num_hidden[1], kernel_size=window_size, padding=int((self.window_size-1)/2))
        # h = F.relu(h)
        # h = F.dropout(h, p=self.dropout, training=self.training)

        h = h.squeeze().T
        arg_index = torch.argsort(sort_index) # 该操作将恢复排序前的序列
        h = h[arg_index]

        final_h = torch.cat([before_h, h], dim=1)
        # h 的 num_nodes 数量不会减少。这是因为在定义一维卷积层时，指定了 padding=int((self.window_size-1)/2) 参数。
        # 这个参数会在输入数据的两端填充一定数量的零，以保证卷积后输出数据的大小不变
        # final_h 的形状是 (batch_size, num_nodes, 2 * num_hidden[1])
        final_h = self.final_layer(final_h)
        # 最后变成(batch_size,num_nodes，num_class)
        # 如果只有一张图，batch_size就是1
        return F.log_softmax(final_h, 1)
        # 概率分布 shape值不变，只不过把num_class特征变成概率
