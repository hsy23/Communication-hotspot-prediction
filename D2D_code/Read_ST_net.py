import gc
import argparse
import dgl
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings

"""Multi-channel temporal end-to-end framework"""

"""Input datasets
    dgls: dataframe with userid as index, contains labels, and activity sequence if needed
    macro: dataframe containing macroscopic data if needed 宏观数据
    graphs: dictionary with format {user_id: list of networkx graphs}}

    ** Modify acitivity and macro flags to run different versions of model to include features
"""


def get_graph(arg, dgls):
    for g_seq_list in arg:  # 时间图序列
        temp_g = []
        for g in g_seq_list[:len(g_seq_list)]:
            G = dgl.DGLGraph(g)
            temp_g.append(G)
        dgls.append(temp_g)
    print('read graphs done')
    return dgls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=int, default=9)  # 8 periods for train,Time Window is about 1 week
    parser.add_argument('--label_path', type=str, default='./spatio-temporal net/cate_data_del0_hott/app_label_2.npy')
    parser.add_argument('--graphs_path', type=str,
                        default='./spatio-temporal net/cate_data_del0_hott/app_sub_net_2.npy')
    parser.add_argument('--input_path', type=str, default='./spatio-temporal net/cate_data_del0_hott/app_feature_2.npy')

    args = parser.parse_args()
    k = args.period

    graphs_path = './spatio-temporal net/cate_data_del1_hott/app_sub_net_'
    input_path = './spatio-temporal net/cate_data_del1_hott/app_feature_'
    label_path = './spatio-temporal net/cate_data_del1_hott/app_label_'

    key_value = []
    per_value = []
    for i in range(28):
        print('Cate:{}'.format(i))
        dgls = []
        # load data
        try:
            with open(label_path + str(i) + '.npy', 'rb') as f:
                labels = np.load(f)
            with open(graphs_path + str(i) + '.npy', 'rb') as f:
                graphs_sep = pickle.load(f)
                dgls = get_graph(graphs_sep, dgls)
        except:
            continue
        n = len(dgls)
        if n == 0:
            continue
        print('size of labels:{}\n, num of dgls = {}'.format(len(labels), n))
        print('num of not hot:{}, hotspots:{}'.format(sum(sum(labels == 0)), sum(sum(labels == 1))))
        print('hot/all:{:.3f}%'.format(sum(sum(labels == 1))/(sum(sum(labels == 0))+sum(sum(labels == 1)))*100))

        nodes_num_list = []
        for j in range(n):
            nodes_num = 0
            for graph in dgls[j]:
                nodes_num += graph.number_of_nodes()
            if nodes_num == 0:
                print(i, j)
            nodes_num_list.append(nodes_num)
        min_v, max_v, avg = min(nodes_num_list), max(nodes_num_list), np.mean(nodes_num_list)
        p25, p50, p75 = np.percentile(nodes_num_list, 25), np.percentile(nodes_num_list, 50), \
                        np.percentile(nodes_num_list, 75),

        key_value.append([min_v, max_v, avg])
        per_value.append([p25, p50, p75])
        print(min_v, max_v, avg)
        print(p25, p50, p75, '\n')

    # file1 = 'D2D__del1_keyinfo'
    # file2 = 'D2D__del1_perinfo'
    # np.save(file1, key_value, allow_pickle=True)
    # np.save(file2, per_value, allow_pickle=True)
