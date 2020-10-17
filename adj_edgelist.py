import pandas as pd
import numpy as np

def adj_edgelist():
    """
    return adjacent matrix and edge list
    """
    name = open('./data/name.csv')
    df_name = pd.read_csv(name)
    name_dict = {}

    # label for dict
    # label 739 target nodes, begin from 0
    label_num = 0

    for i in range(len(df_name)):
        name_ = df_name['name_node_pairs'][i]
        name_dict[name_] = label_num
        label_num += 1

    # label other nodes
    for i in range(len(df_name)):
        print(i)
        name_ = df_name['name_node_pairs'][i]
        data = open('./data/source_data/{}.csv'.format(name_))
        df_data = pd.read_csv(data)

        for j in range(len(df_data)):
            x = df_data['From'][j]
            y = df_data['To'][j]
            s = '{}_{}'.format(x, y)
            # make sure x<y
            # 0xa_0xb and 0xb_0xa are the same node
            # otherwise 0xa_0xb and 0xb_0xa is considered to be different nodes
            if x > y:
                s = '{}_{}'.format(y, x)
            if s not in name_dict:
                name_dict[s] = label_num
                label_num += 1

    print('label_num: {}'.format(label_num))
    # total 164828 nodes

    return adj, edgelist, name_dict