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
            y1 = label_dict[i][pos+1:]
            x2 = label_dict[j][0:pos]
            y2 = label_dict[j][pos+1:]
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

def edge_list():
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
    # reverse the dict
    # key: node pair, value: label -> key: label, value: node pair
    label_dict = {v: k for k, v in name_dict.items()}
    f = open('./data/node2vec/edgelist/temp_link_pred_edgelist.txt', 'w')
    for i in range(736):
        pos = label_dict[i].find('_')
        # print('i = {}'.format(i))
        for j in range(i + 1, total_num):

            x1 = label_dict[i][0:pos]
            y1 = label_dict[i][pos+1:]
            x2 = label_dict[j][0:pos]
            y2 = label_dict[j][pos+1:]
            if x1 == x2 or x1 == y2 or y1 == x2 or y1 == y2:
                f.write('{} {}\n'.format(i, j))
    f.close()

def plotGraph(withLabels = False):
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
    nx.draw(G, with_labels=withLabels, node_size=250)
    plt.show()

    # return

def aggregate_neighbors_node2vec():
    """
    two-phase process: aggregate 736 nodes
    For 736 nodes, aggregate every node with its 1-order neighbor.
    aggregation method: node2vec
    output: concatenate node2vec features to original features
    :return:
    """
    # edgelist of nodes
    name = open('./data/name.csv')
    df_name = pd.read_csv(name)
    name_dict = {}

    for ii in range(len(df_name)):

        name_ = df_name['name_node_pairs'][ii]
        print('ii = {}, name = {}'.format(ii, name_))
        file = open('./data/source_data/{}.csv'.format(name_))
        data = pd.read_csv(file)

        n = 0
        for i in range(len(data)):
            x = data['From'][i]
            y = data['To'][i]
            if y<x:
                temp = y
                y = x
                x = temp
            s = x+'_'+y
            if s not in name_dict:
                name_dict[s] = n
                n += 1
        label_dict = {v: k for k, v in name_dict.items()}
        f = open('./data/aggregation/node2vec/edgelist/{}.txt'.format(name_), 'w')
        for i in range(len(label_dict)):
            pos = label_dict[i].find('_')
            for j in range(i+1, len(label_dict)):
                x1 = label_dict[i][0:pos]
                y1 = label_dict[i][pos + 1:]
                x2 = label_dict[j][0:pos]
                y2 = label_dict[j][pos + 1:]
                if x1 == x2 or x1 == y2 or y1 == x2 or y1 == y2:
                    f.write('{} {}\n'.format(i, j))
        f.close()

def find_label():
    """
    for every node pair's edgelist, find its label
    :return:
    """
    # edgelist of nodes
    name = open('./data/name.csv')
    df_name = pd.read_csv(name)
    name_dict = {}

    label = pd.DataFrame(columns=['name_node_pairs', 'label'])
    for ii in range(len(df_name)):

        name_ = df_name['name_node_pairs'][ii]
        print('ii = {}, name = {}'.format(ii, name_))
        file = open('./data/source_data/{}.csv'.format(name_))
        data = pd.read_csv(file)
        pos_ = name_.find('_')
        x_ = name_[0:pos_]
        y_ = name_[pos_+1:]
        if x_>y_:
            temp = y_
            y_ = x_
            x_ = temp
        name_ = x_ + '_' + y_
        n = 0

        alreadyHave = False

        for i in range(len(data)):
            x = data['From'][i]
            y = data['To'][i]
            if y<x:
                temp = y
                y = x
                x = temp
            s = x+'_'+y
            if s == name_ and not alreadyHave:
                label = label.append([{'name_node_pairs': name_, 'label': n}], ignore_index=True)
                alreadyHave = True
            if s not in name_dict:
                name_dict[s] = n
                n += 1

    label.to_csv('./data/aggregation/node2vec/edgelist/label.csv', index=False)

def aggregate_neighbors_node2vec_node():
    """
    two-phase process: 736 times, concatenate two nodes' vector, without transforming to line graph
    :return:
    """
    # edgelist of nodes
    name = open('./data/name.csv')
    df_name = pd.read_csv(name)
    name_dict = {}
    label = pd.DataFrame(columns=['name_node_pairs', 'x', 'y'])
    for ii in range(len(df_name)):
        print('ii = {}'.format(ii))
        name_ = df_name['name_node_pairs'][ii]
        print('ii = {}, name = {}'.format(ii, name_))
        file = open('./data/source_data/{}.csv'.format(name_))
        data = pd.read_csv(file)

        edges = set()
        n = 0
        f = open('./data/aggregation/node2vec/edgelist_node/{}.txt'.format(name_), 'w')
        for i in range(len(data)):
            x = data['From'][i]
            y = data['To'][i]
            if x not in name_dict:
                name_dict[x] = n
                n += 1
            if y not in name_dict:
                name_dict[y] = n
                n += 1
            str2write = '{} {}\n'.format(name_dict[x], name_dict[y])
            if str2write not in edges:
                f.write(str2write)
        pos = name_.find('_')
        x_ = name_[0:pos]
        y_ = name_[pos+1:]
        label = label.append([{'name_node_pairs': name_, 'x': name_dict[x_], 'y': name_dict[y_]}], ignore_index=True)
        f.close()
    label.to_csv('./data/aggregation/node2vec/edgelist_node/label.csv', index=False)

def aggregation_node2vec_cmd():
    """
    run node2vec for all node pairs
    :return:
    """
    name = open('./data/name.csv')
    df_name = pd.read_csv(name)
    f = open('./data/aggregation/node2vec/edgelist_node/node2vec_cmd.txt', 'w')
    for i in range(len(df_name)):
        name_ = df_name['name_node_pairs'][i]
        f.write('python -m openne --method node2vec --input data/temp_link_pred/{}.txt  --graph-format edgelist --output output/vec_{}.txt --q 0.25 --p 0.25\n'.format(name_, name_))
    f.close

def readEmbedding(rootdir):
    f = open(rootdir)
    line = f.readline()
    data_array = []
    while line:
        num = list(map(float, line.split(' ')))
        data_array.append(num)
        line = f.readline()
    f.close()
    # 736 128
    # 0 x x x x x x x x
    # 1 x x x x x x x x
    # 2 ......
    del data_array[0] # delete the first row
    # data_array = list(map(lambda x:x[1:], data_array)) # delete the first column
    data_array = np.array(data_array)
    return data_array

def concat_node2vec():
    name = open('./data/name.csv')
    df_name = pd.read_csv(name)
    label = open('./data/aggregation/node2vec/edgelist_node/label.csv')
    df_label = pd.read_csv(label)
    for i in range(len(df_name)):
        print('i = {}'.format(i))
        name_ = df_name['name_node_pairs'][i]
        vec = readEmbedding('./data/aggregation/node2vec/outputvec/vec_{}.txt'.format(name_))
        x = df_label['x'][i]
        y = df_label['y'][i]

        feature = open('./data/features_4/{}_temp_link_ft.csv'.format(name_))
        feature_ = pd.read_csv(feature)

        vec_ = np.array([])
        for j in range(len(vec)):
            if vec[j, 0] == x:
                vec_ = np.concatenate([vec_, vec[j, 1:]])
            if vec[j, 0] == y:
                vec_ = np.concatenate([vec_, vec[j, 1:]])
        for j in range(len(vec_)):
            feature_['node2vec_{}'.format(j)] = [vec_[j] for k in range(len(feature_))]

        feature_.to_csv('./data/features_4_node2vec/{}_temp_link_ft.csv'.format(name_), index=False)


# def aggregate_neighbors_GCN():
# python -m openne --method node2vec --input data/temp_link_pred/0x564286362092d8e7936f0549571a803b203aaced_0xf2b1fdc974a80ae077f285421eb39e10403fb1f2.txt  --graph-format edgelist --output vec_temp_link_pred_node2vec.txt --q 0.25 --p 0.25

if __name__ == '__main__':
    print('preprocess: ')
    # adj_edgelist()
    # plotGraph(withLabels=False)
    # edge_list()
    # aggregate_neighbors_node2vec()
    # find_label()
    # aggregate_neighbors_node2vec_node()
    # aggregation_node2vec_cmd()
    concat_node2vec()