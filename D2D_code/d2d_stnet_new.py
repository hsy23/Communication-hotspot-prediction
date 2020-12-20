''''
Data pre-process:
implement of D2D record to adjacency_list
implement of sub-net sampling through adjacency_list
'''
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from D2D_code.Geohash import gps_encode

ghash_code_index = 'bcfguvyz89destwx2367kmqr0145hjnp'
ghash_num_index = range(32)
geo_dict = dict(zip(ghash_code_index, ghash_num_index))
d2d_net_key = np.load("D2D__del1_keyinfo.npy")  # min max avg等关键值
d2d_net_per = np.load("D2D__del1_perinfo.npy")  # 百分位值
D2D_graph_num = np.zeros(28)
P = 0.5


def oneapp_list(use_data):
    """
    :param use_data:同一app类别下的全部D2D记录
    :return: G(networkx graph),record_time,record_gps
    """

    user_id = use_data[:, [1, 2]]
    time = use_data[:, 3].astype(np.float)
    gps = gps_encode(list(use_data[:, 4]))  # 对地理位置进行geohash编码
    G = nx.DiGraph()

    # 将用户名转化为id标识,并生成D2D记录邻接表
    j = 0  # 用来记录record的id
    ud2rc = {}  # user_id to record
    for u1, u2 in user_id:
        ud2rc.setdefault(u1, []).append(j)
        ud2rc.setdefault(u2, []).append(j)
        j += 1

    j = 0
    for u1, u2 in user_id:
        if u1 in ud2rc and len(ud2rc[u1]) > 0:  # u1在之前出现过
            for rc in ud2rc[u1]:  # u1参与的所有记
                if time[rc] > time[j]:
                    G.add_edge(j, rc)
                else:
                    G.add_edge(rc, j)

        if u2 in ud2rc and len(ud2rc[u2]) > 0:  # u2在之前出现过
            for rc in ud2rc[u2]:
                if time[rc] > time[j]:
                    G.add_edge(j, rc)
                else:
                    G.add_edge(rc, j)

        j = j + 1

    return G, time, gps


app_sub_net_path = 'spatio-temporal net/d2d_data_delavg_hott/app_sub_net_'
app_feature_path = 'spatio-temporal net/d2d_data_delavg_hott/app_feature_'
app_label_path = 'spatio-temporal net/d2d_data_delavg_hott/app_label_'


# def net_complete():  # 读取数据并调用其他功能函数进行处理
#     data_path = 'C:/Users/Administrator/Anaconda3/envs/pytorch/projects/D2D Diffusion Prediction/D2D-prediction/full.csv'
#     app_dict_p = 'app_dict_last.txt'
#     time_index_p = 'time_attribute.txt'
#     cate_dict_p = 'cate_dict.csv'
#
#     with open(app_dict_p, 'r', encoding='UTF-8') as f:
#         app_dict = f.read()  # 完整app分类字典
#         app_dict = dict(eval(app_dict))
#
#     with open(time_index_p, 'r', encoding='UTF-8') as time_index:
#         time_partition = time_index.readlines()
#
#     cate_dict = pd.read_csv(cate_dict_p)
#     row_num = list(app_dict.values())
#     for index, cate_row in cate_dict.iterrows():
#         cate_sub_net_path = app_sub_net_path + str(index) + '.npy'
#         cate_feature_path = app_feature_path + str(index) + '.npy'
#         cate_label_path = app_label_path + str(index) + '.npy'
#         cate_save_path = [cate_sub_net_path, cate_feature_path, cate_label_path]
#         if type(cate_row['app id']) is str:
#             app_id_list = cate_row['app id'].strip('{}').split(',')
#         else:
#             app_id_list = list(cate_row['app id'])
#         use_data_cate = pd.DataFrame(columns=('name', 'send_id', 'receive_id', 'time', 'gps'))
#         time_index_cate = [0 for x in range(10)]
#         for i in app_id_list:
#             i = int(i)
#             if i == len(row_num) - 1:  # 区分是不是最后一类app
#                 raw_data = pd.read_csv(data_path, header=None, index_col=False, skiprows=row_num[i][1])
#             else:
#                 raw_data = pd.read_csv(data_path, header=None, index_col=False, skiprows=row_num[i][1],
#                                        nrows=(row_num[i + 1][1] - row_num[i][1]))
#             time_index = time_partition[i].strip('\n').split(',')
#             time_index.remove(time_index[0])  # 去除时间标签的最小值
#             for j in range(10):
#                 time_index_cate[j] += float(time_index[j])
#             use_data = raw_data.iloc[:, [1, 7, 8, 9, 12]].drop_duplicates(keep='first')  # 去除完全重复的记录
#             use_data_cate = pd.concat([use_data_cate, use_data])
#
#         time_index_cate = [x/len(app_id_list) for x in time_index_cate]  # 对时间戳做app平均化处理
#         print('read done:{}'.format(index))
#         print('num of record lines:{}'.format(len(use_data_cate)))
#         use_data = np.array(use_data_cate.values).astype(np.str)
#         one_app = oneapp_list(use_data)
#         sub_net_sampling(one_app, time_index_cate, cate_save_path, index)


def net_complete():  # 读取数据并调用其他功能函数进行处理
    data_path = 'C:/Users/Administrator/Anaconda3/envs/pytorch/projects/D2D Diffusion Prediction/D2D-prediction/full.csv'
    app_dict_p = 'app_dict_last.txt'
    time_index_p = 'time_attribute.txt'
    cate_dict_p = 'cate_dict.csv'

    with open(app_dict_p, 'r', encoding='UTF-8') as f:
        app_dict = f.read()  # 完整app分类字典
        app_dict = dict(eval(app_dict))

    cate_dict = pd.read_csv(cate_dict_p)
    row_num = list(app_dict.values())
    for index, cate_row in cate_dict.iterrows():
        cate_sub_net_path = app_sub_net_path + str(index) + '.npy'
        cate_feature_path = app_feature_path + str(index) + '.npy'
        cate_label_path = app_label_path + str(index) + '.npy'
        cate_save_path = [cate_sub_net_path, cate_feature_path, cate_label_path]
        if type(cate_row['app id']) is str:
            app_id_list = cate_row['app id'].strip('{}').split(',')
        else:
            app_id_list = list(cate_row['app id'])
        use_data_cate = pd.DataFrame(columns=('name', 'send_id', 'receive_id', 'time', 'gps'))
        time_index_cate = []
        for i in app_id_list:
            i = int(i)
            if i == len(row_num) - 1:  # 区分是不是最后一类app
                raw_data = pd.read_csv(data_path, header=None, index_col=False, skiprows=row_num[i][1])
            else:
                raw_data = pd.read_csv(data_path, header=None, index_col=False, skiprows=row_num[i][1],
                                       nrows=(row_num[i + 1][1] - row_num[i][1]))
            use_data = raw_data.iloc[:, [1, 7, 8, 9, 12]].drop_duplicates(keep='first')  # 去除完全重复的记录
            use_data_cate = pd.concat([use_data_cate, use_data])

        time_attr = use_data_cate['time']
        min = time_attr.min()
        max = time_attr.max()
        for j in np.linspace(min, max, 11):
            time_index_cate.append(np.float(j))
        time_index_cate.remove(time_index_cate[0])  # 去掉第一个值
        print('read done:{}'.format(index))
        print('num of record lines:{}'.format(len(use_data_cate)))
        use_data = np.array(use_data_cate.values).astype(np.str)
        one_app = oneapp_list(use_data)
        sub_net_sampling(one_app, time_index_cate, cate_save_path, index)


def sub_net_sampling(args, time_index, save_path, cate_index):  # 将一个（类）app的数据化成时空图网络
    net = args[0]
    app_time = np.array(args[1])
    app_gps = np.array(args[2])
    node_num = len(net.nodes)  # 网络的连通子图数目
    app_sub_net = []
    app_feature = []
    app_label = []
    block = d2d_net_key[cate_index][-1]
    # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。
    # chosen = np.sort(random.sample(net.nodes(), min(700000, int(node_num))))
    sub_net_nodes = 0
    for i in net.nodes():  # 以i为中心的一个子图，产生K个序列子图
        nodes = list(nx.bfs_tree(net, i))  # 通过BFS获取相邻节点列表
        sub_net_nodes += len(nodes)
        for j in reversed(nodes):  # 去除不满足地理位置条件的点
            if app_gps[j][:5] != app_gps[i][:5]:
                nodes.remove(j)
        if len(nodes) <= block:  # 如果获取的节点数小于n，则舍去该子网
            continue

        sequence_net = []
        sequence_feature = []
        sequence_label = []
        for j in range(len(time_index)):  # 去除不满足时间条件的点
            sub_nodes = nodes.copy()
            for t in reversed(sub_nodes):  # 这里使用倒序循环，正序会出现顺序异常
                if app_time[t] > time_index[j] or (app_time[t] < time_index[j-1] if j != 0 else False):
                    sub_nodes.remove(t)
            one_period_label = np.zeros(32)
            one_period_feature = np.zeros((len(sub_nodes), 32))
            for index, t in enumerate(sub_nodes):
                if j == len(time_index) - 1:
                    one_period_label[geo_dict[app_gps[t][-1]]] += 1
                one_period_feature[index, geo_dict[app_gps[t][-1]]] = 1

            one_period_net = net.subgraph(sub_nodes)  # 获得包含这些结点的子网
            mapping = dict(zip(one_period_net, range(len(one_period_net.nodes()))))  # 对节点重新编号从0到n
            one_period_net = nx.relabel_nodes(one_period_net, mapping)

            sequence_net.append(one_period_net)
            sequence_feature.append(one_period_feature)
            if j == len(time_index) - 1:  # 记录最后一次的gps情况作为标签
                thr = one_period_label.mean() + \
                      (one_period_label.max() - one_period_label.mean()) * P
                thr = max(thr, 1)
                one_period_label = np.int64(one_period_label >= thr)  # thr个记录以上定义为热区
                sequence_label.append(one_period_label)

        app_sub_net.append(list(sequence_net))
        app_feature.append(list(sequence_feature))
        app_label.append(sequence_label[0])
    if len(app_sub_net) != 0:
        print('st-graphs num:{}'.format(len(app_feature)))
        D2D_graph_num[cate_index] += len(app_feature)
        # read时每次读取一个app的全部子网 每个app下的一个子网是由10个period的子网组成的子网序列
        # print('{}\n,{}\n,{}'.format(save_path[0], save_path[1], save_path[2]))
        with open(save_path[0], 'wb') as f1:
            pickle.dump(app_sub_net, f1)
            f1.close()
        with open(save_path[1], 'wb') as f2:
            np.save(f2, app_feature)
        with open(save_path[2], 'wb') as f3:
            np.save(f3, app_label)

    print('one cate done\n')


def main():
    net_complete()
    print('done')


if __name__ == '__main__':
    main()
    print(D2D_graph_num)




