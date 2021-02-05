import dgl
import torch
from torch import nn
from dgl.base import DGLError
from dgl import function as fn
import torch.nn.functional as F


def gcn_message(edges):
    return {'m': edges.src['h'] * (1 - edges.data['d']).view((-1, 1)) + edges.dst['h']}


def gcn_reduce(nodes):
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


class EW_Conv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(EW_Conv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def apply(self, nodes):
        return {'h': F.relu(self.linear(nodes.data['h']))}

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature[0]
            g.edata['d'] = feature[1]
            g.update_all(gcn_message, gcn_reduce)
            g.apply_nodes(func=self.apply)
            return g.ndata['h']


def max_reduce(nodes):
    return {'h': torch.max(nodes.mailbox['e'], dim=1)[0]}


def sum_reduce(nodes):
    return {'h': torch.sum(nodes.mailbox['e'], dim=1)}


class EE_Conv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False,
                 allow_zero_in_degree=False,
                 activation=None):
        super(EE_Conv, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.activation = activation

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def message(self, edges):
        """The message computation function.
        """
        theta_x = self.theta(edges.src['h'] * edges.data['d'].view((-1, 1)))
        phi_x = self.phi(edges.dst['h'])
        return {'e': theta_x + phi_x}

    def forward(self, g, feat):
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph')

            g.ndata['h'] = feat[0]
            g.edata['d'] = feat[1]
            if not self.batch_norm:
                g.update_all(self.message, max_reduce)
            else:
                g.apply_edges(self.message)
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(fn.copy_e('e', 'e'), max_reduce)
            if self.activation:
                g.ndata['h'] = self.activation(g.ndata['h'])

            return dgl.mean_nodes(g, 'h')


class EEGCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(EEGCN, self).__init__()
        self.ewgcn = EW_Conv(in_feats, hidden_size)
        self.eegcn = EE_Conv(hidden_size, out_feats, activation=F.relu)

        self.dropout = nn.Dropout(0.5)

    def forward(self, g, inputs):
        h = self.ewgcn(g, inputs)
        h = self.eegcn(g, [h, inputs[-1]])
        h = self.dropout(h)

        return h


if __name__ == '__main__':
    pass
