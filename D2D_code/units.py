import dgl
import torch
import pickle
import numpy as np
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
# create graph input


def get_graph(arg, dgls):
    for g_seq_list in arg:  # 时间图序列
        temp_g = []
        for g in g_seq_list[:len(g_seq_list)]:
            G = dgl.DGLGraph(g)
            temp_g.append(G)
        dgls.append(temp_g)
    return dgls


def get_all_input(path, is_graph):
    result = []
    if not is_graph:
        with open(path, 'rb') as f:
            return np.load(f)
    else:
        with open(path, 'rb') as f:
            # graphs = pickle.load(f)
            # return get_graph(graphs, result)
            return pickle.load(f)


def last_value(args):
    return max(args, key=args.count)

k = 9
def collate_fn(data):
    graphs = []
    features = []
    data_labels = []
    for x in data:
        graphs.append(x[0])
        features.append(x[1])
        data_labels.append(x[2])
    graphs = [dgl.batch([dgl.DGLGraph(u[i]) for u in graphs]) for i in range(k)]
    features = [torch.FloatTensor(np.concatenate([inp[i] for inp in features])) for i in range(k)]
    return graphs, features, Variable(torch.FloatTensor(data_labels))


class MyData(data.Dataset):
    def __init__(self, data_graphs, data_features, data_labels):
        self.data_graphs = data_graphs
        self.data_features = data_features
        self.data_labels = data_labels

    def __len__(self):
        return len(self.data_graphs)

    def __getitem__(self, idx):
        return self.data_graphs[idx], self.data_features[idx], self.data_labels[idx]

