import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


def gcn_message(edgs):
    return {'m': edgs.src['h']}


def gcn_reduce(nodes):
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, last=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.last = last

    def apply(self, nodes):
        return {'h': F.relu(self.linear(nodes.data['h']))}  # 这里做了线性变化+relu处理

    def forward(self, block, feature):
        with block.local_scope():  # 将图像操作本地化，不影响原始图形
            h_src = feature
            h_dst = feature[:block.number_of_dst_nodes()]
            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst

            block.update_all(gcn_message, gcn_reduce)
            block.apply_nodes(func=self.apply)
            if self.last:
                return dgl.mean_nodes(block, 'h')
            else:
                return block.dstdata.pop('h')

    def cat(self, g):  # 问题：这个是否使用过？？
        l = dgl.unbatch(g)
        return torch.stack([g.ndata['h'].view(-1) for g in l], 0)  # view函数将tensor排成一行

    def max_pool(self, g):
        l = dgl.unbatch(g)
        return torch.stack([torch.max(g.ndata['h'], 0)[0] for g in l], 0)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size, False)
        self.gcn2 = GCNLayer(hidden_size, hidden_size, True)#两层结构
        self.linear = nn.Linear(hidden_size, num_classes)#映射到类
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, inputs):
        h = self.gcn1(blocks[0], inputs)
        h = self.gcn2(blocks[1], h)

        return self.linear(h)
