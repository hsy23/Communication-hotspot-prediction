'''
gow pre-process:
implement of merged gowalla data to spatial-temporal graph
'''
import time
import random
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from units import get_distance_hav
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

cates = ['Coffee Shop', 'Sandwich Shop', 'Theatre']

gowalla_graph_num = np.zeros(700)
P = 0.01  # for calculating threshold
ghash_code_index = 'bcfguvyz89destwx2367kmqr0145hjnp'
geo_dict = dict(zip(ghash_code_index, range(32)))
friend_path = '../other_data/gowalla/gowalla_friendship.csv'


# Save path
sub_net_path = 'spatio-temporal net/gowalla/sub_net_'
n_feat_path = 'spatio-temporal net/gowalla/n_feat_'
e_feat_path = 'spatio-temporal net/gowalla/e_feat_'
label_path = 'spatio-temporal net/gowalla/label_'


def net_complete():  # Read the data and call other functional functions for processing
    data_path = 'data_sort_by_cate.pkl'  # the data sorted by category after merging data
    merged_data = pd.read_pickle(data_path)
    merged_data = merged_data[['userid', 'lng', 'lat', 't', 'geo']]

    with open("spot_dict.pkl", 'rb') as f:
        spot_id_dict = pickle.load(f)  # The category corresponds to the number of rows that first appear

    cate_keys = list(spot_id_dict.keys())
    row_num = list(spot_id_dict.values())
    for (index, key) in zip(range(len(cate_keys)), cate_keys):
        if key not in cates:
            continue
        cate_sub_net_path = sub_net_path + str(index) + '.pkl'
        cate_n_feat_path = n_feat_path + str(index) + '.pkl'
        cate_e_feat_path = e_feat_path + str(index) + '.pkl'
        cate_label_path = label_path + str(index) + '.pkl'
        cate_save_path = [cate_sub_net_path, cate_n_feat_path, cate_e_feat_path, cate_label_path]
        if index == len(cate_keys) - 1:
            raw_data = merged_data.iloc[row_num[index][1]:]
        else:
            raw_data = merged_data.iloc[row_num[index][1]:row_num[index + 1][1]]

        random.seed(30)
        chosen = np.sort(random.sample(range(len(raw_data)), min(500000, len(raw_data))))
        raw_data = raw_data.iloc[chosen]
        pro_data = raw_data.drop_duplicates(keep='first')  # Remove duplicated records
        pro_data.sort_values(by="geo", inplace=True)
        pro_data = pro_data.reset_index().iloc[:, 1:]

        time_attr = pro_data['t']
        t_min = time_attr.min()
        t_max = time_attr.max()
        time_index = []
        for j in np.linspace(t_min, t_max, 11):
            time_index.append(np.float(j))
        time_index.remove(time_index[0])
        print('read done:{}'.format(index))
        print('num of record lines:{}'.format(len(pro_data)))
        pro_data = np.array(pro_data.values).astype(np.str)
        one_cate = Static_graph(pro_data)
        sub_net_sampling(one_cate, time_index, cate_save_path, index)


def Static_graph(pro_data):
    """
    :param pro_data: All Check records under the same location category (gowalla )
    :return: G(networkx graph),time,gps
    """
    friend_ship = pd.read_csv(friend_path)
    friend_ship = np.array(friend_ship.values)  # 转为numpy数组
    g_tmp = nx.Graph()
    g_tmp.add_edges_from(friend_ship)

    user_id = list(pro_data[:, 0].astype(np.float))
    gps = list(pro_data[:, [2, 1]].astype(np.float))
    time = list(pro_data[:, -2].astype(np.float))
    geo = list(pro_data[:, -1])

    new_g_tmp = g_tmp.subgraph(user_id)

    G = nx.DiGraph()
    G.add_nodes_from(range(len(user_id)))
    # transformed user name into an ID, record the corresponding check-in node
    ud2rc = {}  # user_id to record(node)
    for (index, u) in zip(range(len(user_id)), user_id):
        ud2rc.setdefault(u, []).append(index)

    for u1 in tqdm(new_g_tmp.nodes()):
        ner = list(new_g_tmp.neighbors(u1))
        # Randomly select 35% of friends
        # (https://www.businessinsider.com/35-percent-of-friends-see-your-facebook-posts-2013-8)
        for u2 in random.sample(ner, int(0.35 * len(ner))):
            for node1 in ud2rc[u1]:
                for node2 in ud2rc[u2]:
                    d = get_distance_hav(gps[node1], gps[node2])
                    if time[node1] > time[node2]:
                        G.add_edge(node2, node1, weight=d)
                    else:
                        G.add_edge(node1, node2, weight=d)

    print('Static Graph, node nums:{}, edge nums:{}'.format(G.number_of_nodes(), G.number_of_edges()))

    return G, time, geo


def sub_net_sampling(args, time_index, save_path, cate_index):
    # Converts a category's data into a spatial-temporal graph
    net = args[0]
    nodes_all = list(net.nodes)
    time = np.array(args[1])
    geo = np.array(args[2])
    node_num = len(nodes_all)

    sub_net = []
    node_feature = []
    edge_feature = []
    label = []
    random.seed(23)
    chosen = np.sort(random.sample(net.nodes(), min(20000, int(node_num))))
    # chosen = nodes_all
    dup_dict = {}
    net_index = 0
    for i in tqdm(chosen):  # 以i为中心的一个子图，产生K个序列子图
        if geo[i] in dup_dict.keys():
            continue
        nodes = list(range(max(0, i-1000), min(i+1000, node_num)))
        for j in reversed(nodes):  # 去除不满足地理位置条件的点
            if geo[j][:5] != geo[i][:5]:
                nodes.remove(j)

        dup_dict.setdefault(geo[i], []).append(net_index)
        net_index += 1

        sequence_net = []
        sequence_feature = []
        sequence_edge = []
        sequence_label = []
        for j in range(len(time_index)):  # get rid of the points that don't satisfy the time condition
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
            if j == len(time_index) - 1:  # Record the last for the tag
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


def main():
    net_complete()
    print('done')


if __name__ == '__main__':
    main()





