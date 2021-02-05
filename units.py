import dgl
import torch
import pickle
import numpy as np
from torch.autograd import Variable
from torch.utils import data
from math import sin, asin, cos, radians, fabs, sqrt
from torch.utils.data import DataLoader
# create graph input

k = 9


def get_all_input(path, is_graph):
    if not is_graph:
        with open(path, 'rb') as f:
            return np.load(f)
    else:
        with open(path, 'rb') as f:
            return np.load(f)


def last_value(args):
    return max(args, key=args.count)


def collate_fn(data):
    graphs = []
    n_feat = []
    e_feat = []
    data_labels = []
    for x in data:
        graphs.append(x[0])
        n_feat.append(x[1])
        e_feat.append(x[2])
        data_labels.append(x[3])
    graphs = [dgl.batch([dgl.from_networkx(u[i]) for u in graphs]) for i in range(k)]
    n_feat = [torch.FloatTensor(np.concatenate([inp[i] for inp in n_feat])) for i in range(k)]
    e_feat = [torch.FloatTensor(np.concatenate([inp[i] for inp in e_feat])) for i in range(k)]
    return graphs, n_feat, e_feat, torch.FloatTensor(data_labels)


def collate_fn_old(data):
    graphs = []
    n_feat = []
    e_feat = []
    data_labels = []
    for x in data:
        graphs.append(x[0])
        n_feat.append(x[1])
        e_feat.append(x[2])
        data_labels.append(x[3])
    graphs = [dgl.batch([dgl.DGLGraph(u[i]) for u in graphs]) for i in range(k)]
    n_feat = [torch.FloatTensor(np.concatenate([inp[i] for inp in n_feat])) for i in range(k)]
    e_feat = [torch.FloatTensor(np.concatenate([inp[i] for inp in e_feat])) for i in range(k)]
    return graphs, n_feat, e_feat, torch.FloatTensor(data_labels)


class MyData(data.Dataset):
    def __init__(self, data_graphs, data_n_feat, data_e_feat, data_labels):
        self.data_graphs = data_graphs
        self.data_n_feat = data_n_feat
        self.data_e_feat = data_e_feat
        self.data_labels = data_labels

    def __len__(self):
        return len(self.data_graphs)

    def __getitem__(self, idx):
        return self.data_graphs[idx], self.data_n_feat[idx], self.data_e_feat[idx], self.data_labels[idx]

