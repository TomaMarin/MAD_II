import networkx as nx
import random
import matplotlib.pyplot as plt
# from networkx.generators.random_graphs import _random_subset
import numpy as np
import copy
from networkx.algorithms import community
from networkx.algorithms.community import k_clique_communities, greedy_modularity_communities

facebook_matrix = np.array([
    [0, 0, 0, 1, 0, 1, 0, 0],  # cici
    [0, 0, 1, 0, 0, 0, 0, 0],  # mat
    [0, 1, 0, 0, 0, 0, 0, 0],  # mark
    [1, 0, 0, 0, 0, 1, 0, 0],  # sere
    [0, 0, 0, 0, 0, 0, 0, 0],  # bin
    [1, 0, 0, 1, 0, 0, 0, 0],  # luca
    [0, 0, 0, 0, 0, 0, 0, 0],  # stine
    [0, 0, 0, 0, 0, 0, 0, 0]  # barby
])

linkedin_matrix = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 0],
])

work_matrix = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
])

friend_matrix = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0]
])


def read_data_set_to_tensor(file):
    f = open(file, "r")
    layers = list()
    edges = list()
    number_of_layers = 0
    for x in f:
        if "LAYERS" in x:
            while x != "\n":
                x = f.readline()
                layers.append(x)
        if "EDGES" in x:
            while x != "\n":
                x = f.readline()
                edges.append(x)


tensor = [facebook_matrix, linkedin_matrix, work_matrix, friend_matrix]


def find_degree_of_node(nodeIndex, tensor):
    total_degree = 0
    for i in tensor:
        total_degree += sum(1 for k in i[nodeIndex] if k == 1)
    return total_degree


def calculate_degrees_for_all_nodes(tensor):
    nodes_and_degrees = list()
    for i in range(len(tensor[0])):
        nodes_and_degrees.append((i, find_degree_of_node(i, tensor)))
    return nodes_and_degrees


def calculate_stdv(degrees_centrality):
    return np.std(degrees_centrality)


def find_neighbors_of_node(nodeIndex, tensor):
    total_number_of_neighbors = 0
    neighbors_indexes = list()
    for i in tensor:
        for ik, k in enumerate(i[nodeIndex]):
            if k == 1 and np.isin(ik, neighbors_indexes, invert=True):
                total_number_of_neighbors += 1
                neighbors_indexes.append(ik)
    return total_number_of_neighbors


def calculate_number_of_neighbors_for_all_nodes(tensor):
    nodes_and_neighbors = list()
    for i in range(len(tensor[0])):
        nodes_and_neighbors.append((i, find_neighbors_of_node(i, tensor)))
    return nodes_and_neighbors


def calculate_connective_redundancy_for_node(nodeIndex, tensor):
    degree = find_degree_of_node(nodeIndex, tensor)
    neighborhood = find_neighbors_of_node(nodeIndex, tensor)
    return 1 - (neighborhood / degree)


def calculate_connective_redundancy_for_all(tensor):
    nodes_and_results = list()
    for i in range(len(tensor[0])):
        nodes_and_results.append((i, calculate_connective_redundancy_for_node(i, tensor)))
    return nodes_and_results


def calculate_exclusive_neighborhood_number_for_all_and_dimension(tensor, dimnesionindex):
    nodes_and_results = list()
    for i in range(len(tensor[0])):
        nodes_and_results.append(
            (i, calculate_exclusive_neighborhood_number_for_node_and_dimension(i, tensor, dimnesionindex)))
    return nodes_and_results


def calculate_exclusive_neighborhood_number_for_node_and_dimension(nodeIndex, tensor, dimensionIndex):
    neighbors_indexes = list()
    for il, l in enumerate(tensor[dimensionIndex][nodeIndex]):
        if l == 1 and np.isin(il, neighbors_indexes, invert=True):
            neighbors_indexes.append(il)
    for ii, i in enumerate(tensor):
        if ii != dimensionIndex:
            for ik, k in enumerate(i[nodeIndex]):
                if k == 1 and np.isin(ik, neighbors_indexes):
                    neighbors_indexes.remove(ik)
    return len(neighbors_indexes)


def change_incidence_to_adjacency_matrix(matrix):
    adjacency_matrix = copy.deepcopy(matrix)
    for i in adjacency_matrix:
        if i == 0:
            i = 65356
    return adjacency_matrix


def floydWarshall(matrix):
    dist = copy.deepcopy(matrix)
    for iten, n in enumerate(dist):
        for io, o in enumerate(dist):
            if dist[iten][io] == 0 and iten != io:
                dist[iten][io] = 65356
    for k in range(len(matrix)):

        # pick all vertices as source one by one
        for i in range(len(matrix)):

            # Pick all vertices as destination for the
            # above picked source
            for j in range(len(matrix)):
                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i][j]
                dist[i][j] = min(dist[i][j],
                                 dist[i][k] + dist[k][j]
                                 )
    return dist


def random_walker_matrix(nodeIndex, tensor, actorIndex, steps):
    matrix = tensor[nodeIndex]
    visited_nodes = list()
    actual_actor = copy.deepcopy(matrix[actorIndex])
    for i in range(steps):
        neighbour = [ii for ii, i in enumerate(actual_actor) if i == 1]
        chosen_neighbour = np.random.choice(neighbour)
        visited_nodes.append(chosen_neighbour)
        actual_actor = matrix[chosen_neighbour]
    return visited_nodes


def cross_layer_random_walk(tensor, actor, p, actorIndex, steps):
    layer_index = np.random.randint(0, len(tensor))
    current_layer = tensor[layer_index]
    actual_actor = copy.deepcopy(current_layer[actorIndex])
    # for i in range(steps):


def occupation_centrality_of_node(nodeIndex, tensor):
    pass


def find_multilayer_path_between_nodes(firstNodeIndex, secondNodeIndex, tensor):
    all_paths = list()
    # TODO SPRAVIT MULTILAYER DISTANCE
    # https://books.google.sk/books?id=blCJDAAAQBAJ&pg=PA38&hl=sk&source=gbs_toc_r&cad=4#v=onepage&q&f=false
    # https://www.researchgate.net/profile/Piotr_Brodka2/publication/233871561_Multi-layered_Social_Networks/links/0c96052d3cabc6b95e000000/Multi-layered-Social-Networks.pdf
    # https://pdfs.semanticscholar.org/7666/35f71dc17c220ba280a1eb028c221af3d1ea.pdf


def calculate_relevance_of_node_and_dimension(nodeIndex, dimensionIndex, tensor):
    neighbors_indexes = list()
    for ik, k in enumerate(tensor[dimensionIndex][nodeIndex]):
        if k == 1 and np.isin(ik, neighbors_indexes, invert=True):
            neighbors_indexes.append(ik)
    return len(neighbors_indexes) / find_neighbors_of_node(nodeIndex, tensor)


def calculate_exclusive_relevance_of_node_and_dimension(nodeIndex, dimensionIndex, tensor):
    return calculate_exclusive_neighborhood_number_for_node_and_dimension(nodeIndex, tensor,
                                                                          dimensionIndex) / find_neighbors_of_node(
        nodeIndex, tensor)


def basic_flattening(tensor, with_weight):
    flattened_matrix = tensor[0]
    for i in range(1, len(tensor)):
        flattened_matrix += tensor[i]
    if with_weight == False:
        for il, l in enumerate(flattened_matrix):
            for ik, k in enumerate(flattened_matrix):
                if flattened_matrix[il][ik] > 1:
                    flattened_matrix[il][ik] = 1
    return flattened_matrix


def basic_flattening_for_two_layers(firstIndexLayer, secondIndexLayer, tensor, with_weight):
    flattened_matrix = tensor[firstIndexLayer] + tensor[secondIndexLayer]

    if with_weight == False:
        for il, l in enumerate(flattened_matrix):
            for ik, k in enumerate(flattened_matrix):
                if flattened_matrix[il][ik] > 1:
                    flattened_matrix[il][ik] = 1
    return flattened_matrix


def weighted_flattening_for_2_layers(tensor, firstLayerIndex, secondLayerIndex, theta_matrix):
    # vaha pre prvu vrstvu, vaha medzi 2 vrstvami(asi ak maju nodes v oboch maticiach hranu), vaha pre 2. vrstvu,
    first_layer = copy.deepcopy(tensor[firstLayerIndex])
    first_layer *= theta_matrix[firstLayerIndex][firstLayerIndex]
    second_layer = copy.deepcopy(tensor[secondLayerIndex])
    second_layer *= theta_matrix[secondLayerIndex][secondLayerIndex]
    inter_layer = copy.deepcopy(tensor[firstLayerIndex])
    inter_layer.fill(0)

    for i in range(len(tensor[firstLayerIndex])):
        for j in range(len(tensor[firstLayerIndex])):
            if tensor[firstLayerIndex][i][j] == 1 and tensor[secondLayerIndex][i][j] == 1:
                inter_layer[i][j] = 1 * theta_matrix[firstLayerIndex][secondLayerIndex]

    return first_layer + inter_layer + second_layer


def girvan_newman(graph):
    pos = nx.spring_layout(graph)
    comms = community.girvan_newman(graph)
    nodes_of_comms = next(comms)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_comms[0], node_color='r', alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_comms[1], node_color='b', alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_comms[2], node_color='g', alpha=0.8)
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), alpha=0.5, )
    nx.draw_networkx_labels(graph, pos)


def incidence_matrix_to_graph(matrix):
    G = nx.empty_graph()
    for i in range(0, len(matrix)):
        G.add_node(i)

    for ii, i in enumerate(matrix):
        for j in range(len(i)):
            if i[j] == 1:
                G.add_edge(ii, j)
                G.add_edge(j, ii)
    return G


def clique_community_detection(graph, min_size_of_clique):
    return list(k_clique_communities(graph, min_size_of_clique))


def modular_community_detection(graph):
    return greedy_modularity_communities(graph)


# nodes_and_degrees = calculate_degrees_for_all_nodes(tensor)
# std = calculate_stdv([i[1] for i in nodes_and_degrees])
# nodes_and_neighbors = calculate_number_of_neighbors_for_all_nodes(tensor)
# connective_redundancy_of_nodes = calculate_connective_redundancy_for_all(tensor)
# xneighborhoodforD0 = calculate_exclusive_neighborhood_number_for_all_and_dimension(tensor, 0)
# shortest_path_matrix = floydWarshall(linkedin_matrix)
# calculate_relevance_of_node_and_dimension(7, 0, tensor)
# theta_matrix = np.array([[2, 2, 1, 1], [2, 2, 2, 1], [2, 2, 1, 1], [2, 2, 1, 1]])
# weighted_flattened_layer = weighted_flattening_for_2_layers(tensor, 1, 2, theta_matrix)
# read_data_set_to_tensor("friendfeed_ita.mpx")

flattened_matrix = basic_flattening(tensor, False)
graph = incidence_matrix_to_graph(flattened_matrix)
# girvan_newman(graph)
comm = clique_community_detection(graph, 4)
comm1 = greedy_modularity_communities(graph)
# print(flattened_matrix)
print(comm1)
print("ALL DONE")
