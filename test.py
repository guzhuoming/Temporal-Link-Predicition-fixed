# # 看时间是否在那35天内
# import pandas as pd
#
# name = open('./data/name.csv')
# df_node_pairs = pd.read_csv(name)
# _maxTimeStamp = 0
# _minTimeStamp = 9999999999999999999999
#
# for i in range(len(df_node_pairs)):
#     node_pair = df_node_pairs['name_node_pairs']
#     file = open('./data/source_data/{}.csv'.format(node_pair[i]))
#     df = pd.read_csv(file)
#
#     _timestamp = df['TimeStamp'].values
#     if _minTimeStamp>min(_timestamp):
#         _minTimeStamp = min(_timestamp)
#     if _maxTimeStamp<max(_timestamp):
#         _maxTimeStamp = max(_timestamp)
#
# # print('min timestamp:{}'.format(_minTimeStamp)) # 1512662448  2017-12-08 00:00:48
# # print('max timestamp:{}'.format(_maxTimeStamp)) # 1515686397  2018-01-11 23:59:57


# # 将节点对的名字按升序排好
# import pandas as pd
# name = open('./data/name.csv')
# df_node_pairs = pd.read_csv(name)
#
# for i in range(len(df_node_pairs)):
#     node_pair = df_node_pairs['name_node_pairs']
#     s = node_pair[i]
#     pos = s.find('_')
#     x = s[0:pos]
#     y = s[pos+1:]
#     # ft_4 = open('./data/features_4/{}_temp_link_ft.csv'.format(s))
#     # df_ft_4 = pd.read_csv(ft_4)
#     # ft_10 = open('./data/features_10/{}_temp_link_ft.csv'.format(s))
#     # df_ft_10 = pd.read_csv(ft_10)
#     # df_ft_10.drop(['mean_min_dif_curmean_max_dif_past'], axis=1, inplace=True)
#     # data = open('./data/source_data/{}.csv'.format(s))
#     # df_data = pd.read_csv(data)
#     if x > y:
#         s = y+'_'+x
#         # df_data.to_csv('./data/source_data_/{}.csv'.format(s), index=False)
#         df_node_pairs['name_node_pairs'][i] = s
#         # df_ft_4.to_csv('./data/features_4_/{}_temp_link_ft.csv'.format(s), index=False)
#         # df_ft_10.to_csv('./data/features_10_/{}_temp_link_ft.csv'.format(s), index=False)
#
#     # else:
#         # df_data.to_csv('./data/source_data_/{}.csv'.format(s), index=False)
#         # df_ft_4.to_csv('./data/features_4_/{}_temp_link_ft.csv'.format(s), index=False)
#         # df_ft_10.to_csv('./data/features_10_/{}_temp_link_ft.csv'.format(s), index=False)
#
#     df_node_pairs.to_csv('./data/name_sorted.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# file = open('./data/name_duplicate.csv')
# df = pd.read_csv(file)
# df1 = df.drop_duplicates(keep=False)
# df2 = df.drop_duplicates(keep='first')
# df3 = df1.append(df2).drop_duplicates(keep=False)
# print(df3['name_node_pairs'])
# df3.to_csv('./data/name_.csv')
# df = df.drop_duplicates()
# df.to_csv('./data/name_drop_duplicate.csv', index=False)
# print(len(df))


# f = open('./data/number of neighbour nodes.txt')
# li = []
# while 1:
#     line = f.readline()
#     if not line:
#         break
#     pos = line.find(':')
#     s = line[pos+2:-1]
#     li.append(int(s))
#
# a = range(736)
# b = li
# neighborsOfNodes = pd.DataFrame({"labelOfNodes": a,
#                                     "neighbourNum": b})
# neighborsOfNodes.to_csv('./data/neighbour_nums.csv', index=False)
# print(li)
# f.close()

"""
plot number of neibours
"""
f = open('./data/neighbour_nums.csv')
df = pd.read_csv(f)

labels = df['labelOfNodes'].values
neighbors = df['neighbourNum'].values

print('mean: {}, median: {}'.format(np.mean(neighbors), np.median(neighbors)))
plt.figure()
plt.plot(labels, neighbors)
plt.xlabel('node label')
plt.ylabel('number of neighbours')
plt.show()
plt.figure()
plt.hist(neighbors, bins=[0, 50, 100, 200, 500, 1000, 2000, 3000, 5000,10000])
plt.xlabel('number of neighbours')
plt.ylabel('number of nodes')
plt.show()
plt.figure()
plt.hist(neighbors, bins=[0, 50, 100, 200, 500, 1000, 2000, 3000, 5000,10000], cumulative=True)
plt.xlabel('number of neighbours(cumulative)')
plt.ylabel('number of nodes')
plt.show()