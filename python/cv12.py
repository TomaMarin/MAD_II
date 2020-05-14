import time
import csv
import networkx as nx
import random
import matplotlib.pyplot as plt
# from networkx.generators.random_graphs import _random_subset
import numpy as np
import copy
import matplotlib.pyplot as plt
from networkx.algorithms import community


def read_data_set_to_tensor(file):
    f = open(file, "r")
    edges = list()
    for x in f:
        edges.append(x)
    return edges


def to_graph(edges, cat_name):
    graph = nx.empty_graph()
    for i in edges:
        edge1, edge2, name = i.strip().split(",")
        if name == cat_name:
            graph.add_edge(edge1, edge2)
    return graph


def binary_list_to_dec(list):
    ar = [int(i) for i in list]
    ar = ar[::-1]
    res = []
    for i in range(len(ar)):
        res.append(ar[i] * (2 ** i))
    return sum(res)


def find_existence_in_layers(tensor, list_of_layers):
    all_nodes = list(tensor.nodes())
    nodes_and_existence = list()
    for i in all_nodes:
        node_existence = list()
        for j in list_of_layers:
            if i in j:
                node_existence.append(1)
            else:
                node_existence.append(0)

        dec_number = binary_list_to_dec(node_existence)
        nodes_and_existence.append((i, dec_number))
    return nodes_and_existence


def girvan_newman(graph):
    # pos = nx.spring_layout(graph)
    comms = community.girvan_newman(graph)
    nodes_of_comms = next(comms)
    # nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_comms[0], node_color='r', alpha=0.8)
    # nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_comms[1], node_color='b', alpha=0.8)
    # nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_comms[2], node_color='g', alpha=0.8)
    # nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), alpha=0.5, )
    # nx.draw_networkx_labels(graph, pos)


def export_nodes_with_existence_number(nodes_and_existence):
    with open('nodes_with_existence.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(nodes_and_existence)


def export_edges(edges):
    with open('edges_of_tensor.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(edges)


edges = read_data_set_to_tensor("friendfeed_ita.mpx")
# to_graph(edges)

like_graph = to_graph(edges, "like")
follow_graph = to_graph(edges, "follow")
comment_graph = to_graph(edges, "comment")
list_of_layers = list()
list_of_layers.append(list(like_graph.nodes()))
list_of_layers.append(list(follow_graph.nodes()))
list_of_layers.append(list(comment_graph.nodes()))
tensor = nx.compose(nx.compose(like_graph, follow_graph), comment_graph)
# girvan_newman(tensor)
# plt.subplot()
# nx.draw(comment_graph, node_size=1)
# plt.show()
# print("pred atributmi")
start = time.time()
nodes_and_existence = find_existence_in_layers(tensor, list_of_layers)
end = time.time()
# print(end-start)
# print(d)
export_nodes_with_existence_number(nodes_and_existence)
export_edges(nx.edges(tensor))
print("done")
