import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def adj_edgelist():
    """
    return adjacent matrix and edge list
    """
    name = open('./data/name.csv')
    df_name = pd.read_csv(name)
    name_dict = {}

    # label for dict
    # label 739 target nodes, begin from 0
    total_num = 0

    for i in range(len(df_name)):
        name_ = df_name['name_node_pairs'][i]
        name_dict[name_] = total_num
        total_num += 1
    """
    count total number of all nodes(node pairs)
    """
    # label other nodes
    for i in range(len(df_name)):
        print(i)
        name_ = df_name['name_node_pairs'][i]
        data = open('./data/source_data/{}.csv'.format(name_))
        df_data = pd.read_csv(data)

        for j in range(len(df_data)):
            x = df_data['From'][j]
            y = df_data['To'][j]
            v = df_data['Value'][j]  # v is used to filter out node pairs whose value is 0
            s = '{}_{}'.format(x, y)
            # make sure x<y
            # 0xa_0xb and 0xb_0xa are the same node
            # otherwise 0xa_0xb and 0xb_0xa will be considered to be different nodes
            if x > y:
                s = '{}_{}'.format(y, x)
            if (s not in name_dict) and (v != 0):
                name_dict[s] = total_num
                total_num += 1

    print('total_num: {}'.format(total_num))

    """
    create edgelist file for node2vec
    """

    # reverse the dict
    # key: node pair, value: label -> key: label, value: node pair
    label_dict = {v: k for k, v in name_dict.items()}

    # save edgelist.txt
    f = open('./data/node2vec/edgelist/temp_link_pred_edgelist.txt', 'w')
    pos = label_dict[0].find('_')

    edge_num = 0

    neighbors_736nodes = [0 for i in range(736)]
    # check the number of neighbours of the 736 nodes
    # delete some unuseful neighbours of some nodes who have too many neighbours
    for i in range(total_num):
        print('i = {}'.format(i))
        for j in range(i + 1, total_num):
            x1 = label_dict[i][0:pos]
            y1 = label_dict[i][pos:]
            x2 = label_dict[j][0:pos]
            y2 = label_dict[j][pos:]
            if x1 == x2 or x1 == y2 or y1 == x2 or y1 == y2:
                f.write('{} {}\n'.format(i, j))
                edge_num += 1
                if i < 736:
                    neighbors_736nodes[i] = neighbors_736nodes[i] + 1

    # for i in range(736):
    #     print('Number of neigbours of node {}: {}'.format(i, neighbors_736nodes[i]))

    """
    save number of neighbours of nodes into dataframe(csv)
    """
    labelOfNodes = range(len(736))

    neighborsOfNodes = pd.DataFrame({"labelOfNodes": labelOfNodes,
                                     "neighbourNum": neighbors_736nodes})
    neighborsOfNodes.to_csv('./data/neighbour_nums')

    f.close()
    print('edge_num: {}'.format(edge_num))

    '''
    number of nodes before filtering: 164824
    number of edges before filtering: 228902020

    number of nodes after filtering: 159833 
    number of edges after filering: 224877458
    '''

    # return adj, edgelist, name_dict

def plotGraph():
    """
    print the graph of source nodes
    calculate the number of different nodes
    :return:
    """
    name = open('./data/name.csv')
    df_name = pd.read_csv(name)

    # calculate number of nodes
    # build dictionary for nodes
    node_dict = {}
    num = 0
    for i in range(len(df_name)):
        name_ = df_name['name_node_pairs'][i]
        pos = name_.find('_')
        x = name_[0:pos]
        y = name_[pos+1:]
        if x not in node_dict:
            node_dict[x] = num
            num += 1
        if y not in node_dict:
            node_dict[y] = num
            num += 1
    print('Number of nodes: {}'.format(len(node_dict)))
    # 581

    # plot the Graph
    G = nx.Graph()
    for i in range(len(df_name)):
        name_ = df_name['name_node_pairs'][i]
        pos = name_.find('_')
        x = name_[0:pos]
        y = name_[pos + 1:]
        G.add_edge(node_dict[x], node_dict[y])
    plt.figure()
    nx.draw(G, with_labels=True)
    plt.show()

    # return



if __name__ == '__main__':
    print('preprocess: ')
    # adj_edgelist()
    # plotGraph()