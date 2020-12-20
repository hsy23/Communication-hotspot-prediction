import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp1
import palettable

# INFOCOM论文用，提升了某些类的占比 不准确
# used_labels = ["GAME:23%","VIDEO_PLAYERS:17%","COMMUNICATION:10%","TOOLS:9%","DATING:7%","ART_AND_DESIGN:7%",
#          "PERSONALIZATION:4%","BUSINESS:4%","FINANCE:4%","FAMILY:3%","PRODUCTIVITY:2%","OTHER 17 CATEGORIES:11%"]
labels = []
lens = 10456231
num_lines_d2d = [2335025, 671963, 407177, 858836, 416909, 328005, 16424, 81311, 361315, 128361, 52400, 127875, 29525,
                 82589, 51281, 856101, 89279, 1705644, 238019, 19163, 79949, 256824, 971147, 94526, 2555, 63870, 31908,
                 98250]


def net_complete_d2d():  # 读取数据并调用其他功能函数进行处理
    data_path = 'full.csv'
    app_dict_p = 'app_dict_last.txt'
    cate_dict_p = 'cate_dict.csv'
    cate_dict = pd.read_csv(cate_dict_p)
    # with open(app_dict_p, 'r', encoding='UTF-8') as f:
    #     app_dict = f.read()  # 完整app字典
    #     app_dict = dict(eval(app_dict))
    #
    # row_num = list(app_dict.values())
    # cate_dict = pd.read_csv(cate_dict_p)
    #
    # all_data = pd.read_csv(data_path)
    # lens = len(all_data)
    # print(lens)
    #
    # for index, cate_row in cate_dict.iterrows():
    #     if type(cate_row['app id']) is str:
    #         app_id_list = cate_row['app id'].strip('{}').split(',')
    #     else:
    #         app_id_list = list(cate_row['app id'])
    #
    #     lins_num = 0
    #
    #     for i in app_id_list:
    #         i = int(i)
    #         if i == len(row_num) - 1:  # 区分是不是最后一类app
    #             lins_num += lens - row_num[i][1]
    #         else:
    #             lins_num += row_num[i + 1][1] - row_num[i][1]
    #     num_lines_d2d.append(lins_num)
    # cate_dict = pd.read_csv(cate_dict_p)
    labels = cate_dict['cate_name']
    # print(num_lines_d2d)
    return labels


def net_complete_lbsn():  # 读取数据并调用其他功能函数进行处理
    with open('spot_dict.txt', 'r', encoding='UTF-8') as f:
        spot_dict = f.read()  # 完整app分类字典
        spot_dict = dict(eval(spot_dict))

    # labels = list(spot_dict.keys())
    row_num = list(spot_dict.values())
    for index in range(len(row_num)):
        if index == len(row_num) - 1:
            lines_num = 6269134 - row_num[index][1]
        else:
            lines_num = row_num[index + 1][1] - row_num[index][1]

        num_lines.append(lines_num)


# def draw_pie():
#     # 设置输出文字类型
#     mp1.rcParams['font.family'] = 'STFangsong'
#     explode = [0.001] * 12
#     print(num_lines, labels)
#     sort_lines, sort_labels = zip(*sorted(zip(num_lines, labels), reverse=True))
#     last_l = list(sort_lines[0:11])
#     last_labels = list(sort_labels[0:11])
#     last_l.append(sum(sort_lines[11:]))
#     # 画面积图
#     colors = list(palettable.cmocean.diverging.Curl_20.mpl_colors[2:14])
#     # colors = list(palettable.cartocolors.qualitative.Prism_10.mpl_colors)
#     colors[-1] = '#BC8F8F'
#     plt.pie(last_l, explode=explode, radius=0.6, colors=colors, wedgeprops={'linewidth': 0.5, 'edgecolor': "black"})
#
#     legend_font = {"family": "Times New Roman", "size":10}
#     plt.legend(used_labels, loc=6,  ncol=1, prop=legend_font)
#     # 可视化呈现
#     plt.show()

def draw_pie(num_lines):
    # 设置输出文字类型
    mp1.rcParams['font.family'] = 'STFangsong'
    explode = [0.001] * 28
    sort_lines, sort_labels = zip(*sorted(zip(num_lines, labels), reverse=True))
    new_label = []
    for i in range(len(sort_labels)):
        print('{},{},{:.2f}%'.format(sort_labels[i], sort_lines[i], sort_lines[i]/lens*100))
        new_label.append(str(sort_labels[i]) + '' + str(round(sort_lines[i]/lens*100, 2)) + '%')
    # last_l = list(sort_lines[0:11])
    # last_labels = list(sort_labels[0:11])
    # last_l.append(sum(sort_lines[11:]))
    # 画面积图
    colors = list(palettable.cmocean.diverging.Curl_20.mpl_colors[2:14])
    # colors = list(palettable.cartocolors.qualitative.Prism_10.mpl_colors)
    colors[-1] = '#BC8F8F'
    plt.pie(sort_lines, explode=explode, radius=0.6, colors=colors, wedgeprops={'linewidth': 0.5, 'edgecolor': "black"})
    legend_font = {"family": "Times New Roman", "size":10}
    plt.legend(new_label, loc=6,  ncol=1, prop=legend_font)
    # 可视化呈现
    plt.show()

if __name__ == '__main__':
    # labels = net_complete_d2d()
    # # np.save('cate labels', labels)
    # # np.save('cate numlines', num_lines)
    # # print(len(labels), len(num_lines))
    # draw_pie(num_lines_d2d)
    a = 0.5
    print(int(a))