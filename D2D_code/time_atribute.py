import numpy as np
import pandas as pd


def time_feature(outfile):
    data_path = 'C:/Users/Administrator/Anaconda3/envs/pytorch/projects/D2D Diffusion Prediction/D2D-prediction/full.csv'
    with open('app_dict_last.txt', 'r', encoding='UTF-8') as f:
        app_dict = f.read()  # 完整app分类字典
        app_dict = dict(eval(app_dict))

    row_num = list(app_dict.values())
    time_list_all = []
    for i in range(len(row_num)):
        time_list = []
        if i == len(row_num)-1:
            raw_data = pd.read_csv(data_path, header=None, index_col=False, skiprows=row_num[i][1])
        else:  # 区分是不是最后一类app
            raw_data = pd.read_csv(data_path, header=None, index_col=False, skiprows=row_num[i][1],
                                   nrows=(row_num[i+1][1]-row_num[i][1]))

        print('read done:{}'.format(i))
        time_attr = raw_data.iloc[:, [9]]
        min = time_attr.min()
        max = time_attr.max()
        dif = max - min  # 计算时间分割
        for j in np.linspace(min, max, 11):
            time_list.append(np.float(j))
        time_list_all.append(time_list)
        for x in time_list:
            print(type(x))
            print('{:.1f}'.format(x))
        # q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # result = list(use_data.quantile(q)[9])  # 分位数，计算占比数。

        # result = [str(i) for i in result]
        #  outfile.write(','.join(result) + '\n')
        break


with open('time_artibute.txt', 'w') as outfile:
    time_feature(outfile)

