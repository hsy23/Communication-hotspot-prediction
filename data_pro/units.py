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
EARTH_RADIUS = 6371  # 地球平均半径，6371km


def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(gps0, gps1):
    """用haversine公式计算球面两点间的距离。"""
    if type(gps0) == np.str:
        try:
            [lat0, lng0] = [float(x) for x in gps0.split('#')]
            [lat1, lng1] = [float(x) for x in gps1.split('#')]
        except:
            return 0.0
    else:
        lat0, lng0, lat1, lng1 = gps0[0], gps0[1], gps1[0], gps1[1]

    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance


def get_graph(arg, dgls):
    for g_seq_list in arg:  # 时间图序列
        temp_g = []
        for g in g_seq_list[:k]:
            G = dgl.DGLGraph(g)
            temp_g.append(G)
        dgls.append(temp_g)
    return dgls


def get_all_input(path, is_graph):
    if not is_graph:
        with open(path, 'rb') as f:
            # return pickle.load(f)
            return np.load(f)
    else:
        with open(path, 'rb') as f:
            # graphs = pickle.load(f)
            # return get_graph(graphs, result)
            # return pickle.load(f)
            return np.load(f)


def last_value(args):
    return max(args, key=args.count)


def collate_fn(data):
    graphs = []
    n_feat = []
    data_labels = []
    for x in data:
        graphs.append(x[0])
        n_feat.append(x[1])
        data_labels.append(x[2])
    graphs = [dgl.batch([dgl.DGLGraph(u[i]) for u in graphs]) for i in range(k)]
    n_feat = [torch.FloatTensor(np.concatenate([inp[i] for inp in n_feat])) for i in range(k)]
    return graphs, n_feat, Variable(torch.FloatTensor(data_labels))


class MyData(data.Dataset):
    def __init__(self, data_graphs, data_n_feat, data_labels):
        self.data_graphs = data_graphs
        self.data_n_feat = data_n_feat
        self.data_labels = data_labels

    def __len__(self):
        return len(self.data_graphs)

    def __getitem__(self, idx):
        return self.data_graphs[idx], self.data_n_feat[idx], self.data_labels[idx]

