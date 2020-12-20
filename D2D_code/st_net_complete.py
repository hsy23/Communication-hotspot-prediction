''''
Data pre-process:
implement of D2D record to adjacency_list
implement of sub-net sampling through adjacency_list
'''
import numpy as np
import pandas as pd
import time
import networkx as nx
import pickle
from D2D_code.Geohash import gps_encode

index = 'bcfguvyz89destwx2367kmqr0145hjnp'
index2 = range(32)
geo_dict = dict(zip(index,index2))

def oneapp_list(use_data):
    '''
    :param use_data:同一app下的全部D2D记录
    :return: G(networkx graph),record_time,record_gps
    '''

    user_id = use_data[:, [1, 2]]
    time = use_data[:, 3].astype(np.int64)
    time_result = []
    gps = gps_encode(list(use_data[:, 4]))  # 对地理位置进行geohash编码
    gps_result = []
    G = nx.DiGraph()

    # 将用户名转化为id标识,并生成D2D记录邻接表
    j = 0  # 用来记录record的id
    num = 0  # 用来记录行数
    ud2rc = {}  # user_id to record
    u_before1 = -1
    u_before2 = -2
    for u1, u2 in user_id:
        # 去除短暂时间内多次发生的的类似记录，保留第一次（最晚发生的）
        try:
            if u1 == u_before1 and u2 == u_before2:
                if gps[num] == gps_result[j-1] and abs(time[num] - time_result[j-1]) < 86400000:
                    num = num + 1   # 该记录跳过
                    continue
        except:
            print(num, j)

        u_before1,u_before2 = u1,u2

        if u1 in ud2rc and len(ud2rc[u1])>0:  # u1在之前出现过
            for rc in ud2rc[u1]:  # u1参与的所有记录
                if time_result[rc] > time[num]:
                    G.add_edge(j, rc)
                else:
                    G.add_edge(rc, j)

        if u2 in ud2rc and len(ud2rc[u2])>0:
            for rc in ud2rc[u2]:
                if time_result[rc] > time[num]:
                    G.add_edge(j, rc)
                else:
                    G.add_edge(rc, j)

        ud2rc.setdefault(u1, []).append(j)
        ud2rc.setdefault(u2, []).append(j)
        time_result.append(time[num])  # 只保留选中的点
        gps_result.append(gps[num])
        j = j+1
        num = num+1

    return (G,time_result,gps_result)

def net_complete():   # 读取数据并调用其他功能函数进行处理
    data_path = 'full.csv'
    app_dict_p = 'app_dict_last.txt'
    time_index_p = 'time_artibute3.txt'

    with open(app_dict_p, 'r', encoding='UTF-8') as f:
        app_dict = f.read()  # 完整app分类字典
        app_dict = dict(eval(app_dict))

    with open(time_index_p,'r',encoding='UTF-8') as time_index:
        time_partition = time_index.readlines()

    row_num = list(app_dict.values())
    for i in range(len(row_num)):
        if i == 10:
            break
        if i == len(row_num)-1:  # 区分是不是最后一类app
            raw_data = pd.read_csv(data_path, header=None, index_col=False,skiprows=row_num[i][1])
        else:
            raw_data = pd.read_csv(data_path, header=None, index_col=False,skiprows=row_num[i][1],nrows=(row_num[i+1][1]-row_num[i][1]))

        time_index = time_partition[i]
        print('read done:{}'.format(i))
        use_data = raw_data.iloc[:, [1, 7, 8, 9, 12]].drop_duplicates(keep='first')  # 去除完全重复的记录
        use_data = np.array(use_data.values).astype(np.str)
        one_app = oneapp_list(use_data)
        sub_net_sampling(one_app,time_index)
        # break


def sub_net_sampling(args, time_index):  # 将一个（类）app的数据化成时空图网络
    time_index = time_index.strip('\n').split(',')
    time_index.remove(time_index[0])  # 去除时间标签的最小值
    net = args[0]
    app_time = np.array(args[1])
    app_gps = np.array(args[2])
    net_num = nx.connected_components(net)  # 网络的连通子图数目
    app_sub_net = []
    app_feature = []
    app_label = []

    if net_num < 5:  # 如果该app网络的连通子图数<5，则跳过
        return
    # chosen = np.sort(random.sample(net.nodes(),min(10000,int(net_num)) )) # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。
    for i in net.nodes():  # 以i为中心的一个子图，产生10个序列子图
        nodes = list(nx.bfs_tree(net, i))  # 通过BFS获取相邻节点列表
        for j in reversed(nodes):  # 去除不满足地理位置条件的点
            if app_gps[j][:5] != app_gps[i][:5]:
                nodes.remove(j)
        if len(nodes) < 5:  # 如果获取的节点数小于5，则舍去该子网
            continue

        sequence_net = []
        sequence_feature = []
        sequence_label = []
        sub_nodes = nodes
        for j in reversed(time_index):  # 去除不满足时间条件的点 这里使用倒序，逐渐删除子结点
            for t in reversed(sub_nodes):  # 这里使用倒序循环，正序会出现顺序异常
                if app_time[t] > float(j):
                    sub_nodes.remove(t)

            one_period_label = np.zeros(32)
            one_period_feature = np.zeros((len(sub_nodes),32))
            for index, t in enumerate(sub_nodes):
                if j == time_index[-1]:
                    one_period_label[geo_dict[app_gps[t][-1]]] += 1
                one_period_feature[index, geo_dict[app_gps[t][-1]]] = 1

            one_period_net = net.subgraph(sub_nodes)  # 获得包含这些结点的子网
            mapping = dict(zip(one_period_net,range(len(one_period_net.nodes()))))  # 对节点重新编号从0到n
            one_period_net = nx.relabel_nodes(one_period_net, mapping)

            sequence_net.append(one_period_net)
            sequence_feature.append(one_period_feature)
            if j == time_index[-1]:  # 记录最后一次的gps情况作为标签
                one_period_label = np.int64(one_period_label > 3)
                sequence_label.append(one_period_label)

        app_sub_net.append(list(reversed(sequence_net)))
        app_feature.append(list(reversed(sequence_feature)))
        app_label.append(sequence_label[0])

    print(len(app_feature))
    if len(app_sub_net) != 0:
        # read时每次读取一个app的全部子网 每个app下的一个子网是由10个perdoid的子网组成的子网序列
        with open('./spatio-temporal net/app_sub_net.npy', 'ab') as f1:
            pickle.dump(app_sub_net, f1)
            f1.close()
        with open("./spatio-temporal net/app_feature.npy", 'ab') as f4:
            np.save(f4, app_feature)
        with open("./spatio-temporal net/app_label.npy", 'ab') as f5:
            np.save(f5, app_label)

    print('one app done')


def main():
    net_complete()
    print('done')


if __name__ == '__main__':
    start = time.clock()
    main()
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)




