import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


def gcn_message(edgs):
    return {'m': edgs.src['h']}


def gcn_reduce(nodes):
    return {'h': torch.sum(nodes.mailbox['m'],dim=1)}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, last=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.last = last

    def apply(self, nodes):
        return {'h': F.relu(self.linear(nodes.data['h']))}  # 这里做了线性变化+relu处理

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_message, gcn_reduce)
        # g.send(g.edges(), gcn_message)
        # g.recv(g.nodes(), gcn_reduce)
        g.apply_nodes(func=self.apply)
        if self.last:
            return dgl.mean_nodes(g, 'h')
        else:
            return g.ndata.pop('h')

    def cat(self, g):  # 问题：这个是否使用过？？
        l = dgl.unbatch(g)
        return torch.stack([g.ndata['h'].view(-1) for g in l], 0)  # view函数将tensor排成一行

    def max_pool(self, g):
        l = dgl.unbatch(g)
        return torch.stack([torch.max(g.ndata['h'], 0)[0] for g in l], 0)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size, False)
        self.gcn2 = GCNLayer(hidden_size, hidden_size, True)  # 两层结构
        self.linear = nn.Linear(hidden_size, out_feats)  # 映射到类
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, inputs):
        inputs = inputs.cuda()
        h = self.gcn1(g, inputs)
        h = self.gcn2(g, h)  # 通过mean_nodes，将维度从所有节点(n)转为1(一个图)

        return self.linear(h)
