"""
4 features:
transaction num, transaction sum, transaction mean, transaction variance
"""
import pandas as pd
import csv
import numpy as np

name = open('./data/name.csv')
df_name = pd.read_csv(name)
name_node_pairs = df_name['name_node_pairs']

# total 12 time periods
ts = 12

# create feature files
for i in range(len(name_node_pairs)):
    file = open('./data/features_4/{}_temp_link_ft.csv'.format(name_node_pairs[i]), 'w',
                newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['tran_num', 'tran_sum', 'tran_mean', 'tran_var'])

    for j in range(ts):
        csvwriter.writerow([0. for i in range(4)])
    file.close()

for i in range(len(name_node_pairs)):
    node_pair = name_node_pairs[i]
    data = open('./data/source_data/{}.csv'.format(node_pair))
    df_data = pd.read_csv(data, index_col=0)
    df_data.index = range(len(df_data))

    # save transaction values
    tran = [[] for i in range(ts)]

    for j in range(len(df_data)):
        npos = node_pair.index('_')
        x = node_pair[0:npos]
        y = node_pair[npos + 1:]
        ft = open('./data/features_4/{}_temp_link_ft.csv'.format(name_node_pairs[i]))
        df_ft = pd.read_csv(ft)
        ft.close()

        if (df_data['From'][j] == x and df_data['To'][j] == y) or \
                (df_data['From'][j] == y and df_data['To'][j] == x):
            t = (df_data['TimeStamp'][j] - 1512576000) // (86400 * 3)
            df_ft['tran_num'][t] = df_ft['tran_num'][t] + 1
            df_ft['tran_sum'][t] = df_ft['tran_sum'][t] + df_data['Value'][j]

            tran[t].append(df_data['Value'][j])

        df_ft.to_csv('./data/features_4/{}_temp_link_ft.csv'.format(name_node_pairs[i]),
                     index=False)

    for t in range(ts):
        if len(tran[t])>0:
            df_ft['tran_mean'][t] = np.mean(tran[t])
            df_ft['tran_var'][t] = np.var(tran[t])

        else:
            df_ft['tran_mean'][t] = 0
            df_ft['tran_var'][t] = 0

    df_ft.to_csv('./data/features_4/{}_temp_link_ft.csv'.format(name_node_pairs[i]),
                 index=False)
