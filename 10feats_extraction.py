'''
extract 10 features:
1. transaction num,
2. transaction sum,
3. transaction mean,
4. transaction variance,
5. the difference between this transaction mean and the min transaction mean of all transactions in the past
6. the difference between this transaction mean and the min transaction mean of all transactions in the current
time period
7. the difference between this transaction mean and the max transaction mean of all transactions in the past
8. the difference between this transaction mean and the max transaction mean of all transactions in the current
time period
9. the difference between this transaction mean and the min transaction mean of all transactions of this node pair
in the past
10. the difference between this transaction mean and the max transaction mean of all transactions of this node pair
in the past
'''

import pandas as pd
import csv

name = open('./data/name.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

ts = 12

for i in range(len(name_node_pairs)):

    file = open('./data/features_10/{}_temp_link_ft.csv'.format(name_node_pairs[i]), 'w',
                newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['tran_num', 'tran_sum', 'tran_mean', 'tran_var', 'mean_min_dif_past', 'mean_min_dif_cur'
                        'mean_max_dif_past', 'mean_max_dif_cur', 'mean_min_dif_past_self_nodepair',
                        'mean_max_dif_past_self_nodepair'])

    for j in range(ts):
        csvwriter.writerow([0. for i in range(10)])
    file.close()

# save transaction mean of 739 node pairs for 12 time periods
tran_avg = []

for i in range(len(name_node_pairs)):
    print('i = {}'.format(i))
    file = open('./data/features_10/{}_temp_link_ft.csv'.format(name_node_pairs[i]))
    df = pd.read_csv(file)
    df.index = range(len(df))

    mean_min_dif_past = []
    mean_min_dif_cur = []
    mean_max_dif_past = []
    mean_max_dif_cur = []
    mean_min_dif_past_self_nodepair = []
    mean_max_dif_past_self_nodepair = []

    for j in range(12):
        if j == 0:
            mean_min_dif_past.append(0)
            mean_min_dif_cur.append(abs(tran_avg[i, j] - min(tran_avg[:, j])))
            mean_max_dif_past.append(0)
            mean_max_dif_cur.append(abs(tran_avg[i, j]-max(tran_avg[:, j])))
            mean_min_dif_past_self_nodepair.append(0)
            mean_max_dif_past_self_nodepair.append(0)
            continue

        temp = tran_avg[:, 0:j].tolist()
        mean_min_dif_past.append(abs(tran_avg[i, j] - min(map(min, temp))))
        mean_min_dif_cur.append(abs(tran_avg[i, j] - min(tran_avg[:, j])))
        mean_max_dif_past.append(abs(tran_avg[i, j] - max(map(max, temp))))
        mean_max_dif_cur.append(abs(tran_avg[i, j] - max(tran_avg[:, j])))
        mean_min_dif_past_self_nodepair.append(abs(tran_avg[i, j] - min(map(min, temp))))
        mean_max_dif_past_self_nodepair.append(abs(tran_avg[i, j]-max(map(max, temp))))

    df.loc[:, 'mean_min_dif_past'] = mean_min_dif_past
    df.loc[:, 'mean_min_dif_cur'] = mean_min_dif_cur
    df.loc[:, 'mean_max_dif_past'] = mean_max_dif_past
    df.loc[:, 'mean_max_dif_cur'] = mean_max_dif_cur
    df.loc[:, 'mean_min_dif_past_self_nodepair'] = mean_min_dif_past_self_nodepair
    df.loc[:, 'mean_max_dif_past_self_nodepair'] = mean_max_dif_past_self_nodepair

    df.to_csv('./data/features_10/{}_temp_link_ft.csv'.format(name_node_pairs[i]), index=False)