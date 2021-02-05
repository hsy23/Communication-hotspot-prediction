import pandas as pd

data = pd.read_csv("googleplaystore_for_cate_class.csv")
cate_data = data.iloc[:, [0, 1]]  # name and  category

app_dict_p = 'app_dict_last.txt'

with open(app_dict_p, 'r', encoding='UTF-8') as f:
    app_dict = f.read()
    app_dict = dict(eval(app_dict))  # app_name_dict

d2d_data = list(app_dict.keys())
category_dict = pd.DataFrame(columns=('category_id', 'cate_name', 'app id'))


def fn_match_num(s1, s2):

    n = len(s1)
    m = len(s2)
    dp = [[0 for i in range(n)] for j in range(m)]
    max_ = 0
    for i in range(m):
        for j in range(n):
            if s2[i] == s1[j]:
                if i > 0 and j > 0:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = 1
                if dp[i][j] > max_:
                    max_ = dp[i][j]
    return max_


c_id = 0
for i, j in enumerate(d2d_data):
    max_math_num = 0
    match_cate = ''
    for index, row in cate_data.iterrows():
        match_num = fn_match_num(j, row['App'])
        if match_num > max_math_num:
            max_math_num, match_cate = match_num, row['Category']
    if match_cate in category_dict['cate_name'].values:
        old = category_dict.loc[category_dict['cate_name'] == match_cate, 'app id'].values
        new = []
        for x in old:
            if type(x) != int:
                for id in x:
                    new.append(id)
            else:
                new.append(x)
        new.append(i)
        category_dict.loc[category_dict['cate_name'] == match_cate, 'app id'] = set(new)
    else:
        category_dict = category_dict.append(pd.DataFrame({'category_id': [c_id], 'cate_name': [match_cate], 'app id': [i]}),
                    ignore_index=True)
        c_id += 1

    print(category_dict)

category_dict.to_csv('OPPST_cate_dict.csv', index=None)
