''''
Data pre-process:
implement of D2D record to adjacency_list
implement of sub-net sampling through adjacency_list
'''
import random
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
from Geohash import gps_encode
from units import get_distance_hav

ghash_code_index = 'bcfguvyz89destwx2367kmqr0145hjnp'
ghash_num_index = range(32)
geo_dict = dict(zip(ghash_code_index, ghash_num_index))
D2D_graph_num = np.zeros(28)
P = 0.01
Block = 100

chosen_cates = [0, 3, 17, 22]

sub_net_path = '../spatio-temporal net/oppst/sub_net_'
n_feat_path = '../spatio-temporal net/oppst/n_feat_'
e_feat_path = '../spatio-temporal net/oppst/e_feat_'
label_path = '../spatio-temporal net/oppst/label_'


def Static_graph(use_data):
    """
    :param use_data:同一app类别下的全部D2D记录
    :return: G(networkx graph),record_time,record_gps
    """

    user_id = use_data[:, [1, 2]]
    time = use_data[:, 3].astype(np.float)
    gps = list(use_data[:, 4])
    geo = gps_encode(gps)  # geohash code
    G = nx.DiGraph()

    j = 0  # 用来记录record的id
    ud2rc = {}
    for u1, u2 in user_id:
        ud2rc.setdefault(u1, []).append(j)
        ud2rc.setdefault(u2, []).append(j)
        j += 1

    j = 0
    for u1, u2 in user_id:
        if u1 in ud2rc and len(ud2rc[u1]) > 0:
            for rc in ud2rc[u1]:
                d = get_distance_hav(gps[j], gps[rc])
                if time[rc] > time[j]:
                    G.add_edge(j, rc, weight=d)
                else:
                    G.add_edge(rc, j, weight=d)

        if u2 in ud2rc and len(ud2rc[u2]) > 0:
            for rc in ud2rc[u2]:
                d = get_distance_hav(gps[j], gps[rc])
                if time[rc] > time[j]:
                    G.add_edge(j, rc, weight=d)
                else:
                    G.add_edge(rc, j, weight=d)

        j = j + 1

    return G, time, geo


def net_complete():
    data_path = 'full.csv'
    app_dict_p = 'app_dict_last.txt'
    cate_dict_p = 'cate_dict.csv'

    with open(app_dict_p, 'r', encoding='UTF-8') as f:
        app_dict = f.read()
        app_dict = dict(eval(app_dict))

    cate_dict = pd.read_csv(cate_dict_p)
    row_num = list(app_dict.values())
    for index, cate_row in cate_dict.iterrows():
        if index not in chosen_cates:
            continue
        cate_sub_net_path = sub_net_path + str(index) + '.pkl'
        cate_n_feat_path = n_feat_path + str(index) + '.pkl'
        cate_e_feat_path = e_feat_path + str(index) + '.pkl'
        cate_label_path = label_path + str(index) + '.pkl'
        cate_save_path = [cate_sub_net_path, cate_n_feat_path, cate_e_feat_path, cate_label_path]
        if type(cate_row['app id']) is str:
            app_id_list = cate_row['app id'].strip('{}').split(',')
        else:
            app_id_list = list(cate_row['app id'])
        use_data_cate = pd.DataFrame(columns=['name', 'send_id', 'receive_id', 'time', 'gps'])
        for i in app_id_list:
            i = int(i)
            if i == len(row_num) - 1:
                raw_data = pd.read_csv(data_path, header=None, index_col=False, skiprows=row_num[i][1])
            else:
                raw_data = pd.read_csv(data_path, header=None, index_col=False, skiprows=row_num[i][1],
                                       nrows=(row_num[i + 1][1] - row_num[i][1]))
            use_data = raw_data.iloc[:, [1, 7, 8, 9, 12]].drop_duplicates(keep='first')
            use_data.columns = ['name', 'send_id', 'receive_id', 'time', 'gps']
            use_data_cate = use_data_cate.append(use_data, ignore_index=True)

        random.seed(23)
        pro_data = use_data_cate
        pro_data.sort_values(by="gps", inplace=True)
        pro_data = pro_data.reset_index().iloc[:, 1:]

        time_attr = pro_data['time']
        t_min = time_attr.min()
        t_max = time_attr.max()
        time_index = []
        for j in np.linspace(t_min, t_max, 11):
            time_index.append(np.float(j))
        time_index.remove(time_index[0])  # remove the smallest
        print('read done:{}'.format(index))
        print('num of record lines:{}'.format(len(pro_data)))
        pro_data = np.array(pro_data.values)
        one_cate = Static_graph(pro_data)
        sub_net_sampling(one_cate, time_index, cate_save_path, index)


def sub_net_sampling(args, time_index, save_path, cate_index):
    net = args[0]
    nodes_all = list(net.nodes)
    time = np.array(args[1])
    geo = np.array(args[2])
    node_num = len(nodes_all)  # 网络的连通子图数目

    sub_net = []
    node_feature = []
    edge_feature = []
    label = []
    random.seed(23)
    chosen = np.sort(random.sample(net.nodes(), min(20000, int(node_num))))
    # chosen = nodes_all
    dup_dict = {}
    net_index = 0
    for i in tqdm(chosen):
        if geo[i] in dup_dict.keys():
            continue
        nodes = list(range(max(0, i-1000), min(i+1000, node_num)))
        for j in reversed(nodes):
            if geo[j][:5] != geo[i][:5]:
                nodes.remove(j)

        dup_dict.setdefault(geo[i], []).append(net_index)
        net_index += 1

        sequence_net = []
        sequence_feature = []
        sequence_edge = []
        sequence_label = []
        for j in range(len(time_index)):
            sub_nodes = nodes.copy()
            for t in reversed(sub_nodes):
                if time[t] > time_index[j] or (time[t] < time_index[j-1] if j != 0 else False):
                    sub_nodes.remove(t)

            one_period_label = np.zeros(32)
            one_period_feature = np.zeros((len(sub_nodes), 32))

            for index, t in enumerate(sub_nodes):
                if j == len(time_index) - 1:
                    one_period_label[geo_dict[geo[t][-1]]] += 1
                one_period_feature[index, geo_dict[geo[t][-1]]] = 1

            one_period_net = net.subgraph(sub_nodes)
            mapping = dict(zip(one_period_net, range(len(one_period_net.nodes()))))
            one_period_net = nx.relabel_nodes(one_period_net, mapping)

            # get d (edge feature)
            attr_tmp = list(one_period_net.edges.values())
            attr_tmp = np.array([x['weight'] for x in attr_tmp])
            if len(attr_tmp) > 0:
                attr_min = attr_tmp.min()
                attr_max = attr_tmp.max()
                if attr_max != 0:
                    print(attr_max)
                attr_tmp = [(x - attr_min) / ((attr_max - attr_min) if (attr_max - attr_min) else 1) for x in attr_tmp]
            sequence_edge.append(attr_tmp)

            # get net and feature
            sequence_net.append(one_period_net)
            sequence_feature.append(one_period_feature)

            # get label
            if j == len(time_index) - 1:
                thr = one_period_label.mean() + \
                      (one_period_label.max() - one_period_label.mean()) * P
                thr = max(thr, 1)

                one_period_label = np.int64(one_period_label >= thr)
                sequence_label.append(one_period_label)

        sub_net.append(sequence_net)
        node_feature.append(sequence_feature)
        edge_feature.append(sequence_edge)
        label.append(sequence_label[0])
    print('dup dict len:', len(dup_dict.values()))

    if len(sub_net) != 0:
        print('st-graphs num:{}, node_feat_list:{}, '
              '\n edge_feat_list:{}, label_list:{}'.format(len(sub_net), len(node_feature), len(edge_feature), len(label)))
        with open(save_path[0], 'wb') as f:
            pickle.dump(sub_net, f)
        with open(save_path[1], 'wb') as f:
            pickle.dump(node_feature, f)
        with open(save_path[2], 'wb') as f:
            pickle.dump(edge_feature, f)
        with open(save_path[3], 'wb') as f:
            pickle.dump(label, f)


if __name__ == '__main__':
    net_complete()
    print('done')




