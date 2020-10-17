# 储存edgelist和邻接矩阵
import pandas as pd

name = open('./data/name.csv')
df_name = pd.read_csv(name)
name_dict = {}
# 从0开始标号，先给前面739个节点标号
label_num = 0
for i in range(len(df_name)):
    name_ = df_name['name_node_pairs'][i]
    name_dict[name_] = label_num
    label_num += 1

for i in range(len(df_name)):
    name_ = df_name['name_node_pairs'][i]
    data = open('./data/source_data/{}.csv'.format(name_))
    df_data = pd.read_csv(data)
    for j in range(len(df_data)):
        x = df_data['From'][j]
        y = df_data['To'][j]
        s = x+'_'+y
        if x>y:
            s = y+'_'+x
