import csv
import time

import numpy as np
from timeit import default_timer as timer
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances, pairwise
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community


def is_number(n):
    try:
        float(n)  # Type-casting the string to `float`.
        # If string is not a valid `float`,
        # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True


def create_net_graph(edges):
    graph = nx.empty_graph()
    for ii, i in enumerate(edges):
        for j in i:
            graph.add_edge(ii, j)
    return graph


def export_edges(edges, fileName):
    with open(fileName, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for ii, i in enumerate(edges):
            for j in i:
                writer.writerow((ii, j))


def export_nodes_with_attributes(dataset, attribute_number, fileName):
    with open(fileName, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for i in range(len(dataset)):
            writer.writerow((i, dataset[i][attribute_number]))


def export(dataset, fileName):
    with open(fileName, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for ii, i in enumerate(dataset):
            if (ii % 2) == 0:
                writer.writerow(i)


def read_data_set(file, delimiter, hasHeader):
    f = open(file, "r")
    dataset = list()
    if hasHeader:
        f.readline()
    for x in f:
        row = x.strip().replace('"', '').split(delimiter)
        row = [float(i) for i in row]
        # for ii, i in enumerate(row):
        #     if is_number(i):
        #         row[ii] = float(i)
        dataset.append(row)
    return dataset


def convert_strings_to_int(dataset):
    for ii, i in enumerate(dataset):
        for ij, j in enumerate(i):
            if is_number(dataset[ii][ij]):
                dataset[ii][ij] = float(dataset[ii][ij])
    return dataset


def calculate_euclid_dist_for_2_objects(object1, object2, ignoreColumns):
    return np.sqrt(sum(pow(a - b, 2) for a, b in zip(object1, object2)))


def create_matrix_by_euclid_dist(dataset, ignoreColumns):
    size = len(dataset)
    matrix = np.zeros((size, size))
    start = timer()
    for ii, i in enumerate(dataset):
        for j in range(ii, size - 1):
            if ii != j:
                result = calculate_euclid_dist_for_2_objects(i, dataset[j], ignoreColumns)
                matrix[ii][j] = result
                matrix[j][ii] = result
    end = timer()
    print("time elapsed: ", end - start)
    return matrix


def knn_and_epsilon_combination_method(matrix, k, epsilon):
    nearest_neighbors = list()
    ie = 0
    ik = 0
    for ii, i in enumerate(matrix):
        e_neighbors = find_neighbours_by_epsilon(i, epsilon, ii)
        if len(e_neighbors) < k:
            nearest_neighbors.append(find_k_nearest_neighbors(i, k, ii))
            ik += 1
        else:
            nearest_neighbors.append(e_neighbors)
            ie += 1
    print("EPSILON: ", ie)
    print("K: ", ik)
    return nearest_neighbors


def find_neighbours_by_epsilon(row, epsilon, rowIndex):
    neighbors = np.argsort(row)
    nearest_neighbors = list()
    # [i for ii, i in enumerate(neighbors) if row[i] < epsilon and i != rowIndex]
    for ii, i in enumerate(neighbors):
        if row[i] < epsilon:
            if i == rowIndex:
                continue
            nearest_neighbors.append(i)
        else:
            break
    return nearest_neighbors


def find_neighbours_by_epsilon_for_matrix(matrix, epsilon):
    nearest_neighbors = list()
    for ii, i in enumerate(matrix):
        nearest_neighbors.append(find_neighbours_by_epsilon(i, epsilon, ii))
    return nearest_neighbors


def find_k_nearest_neighbors(row, k, rowIndex):
    neighbors_indexes = np.argpartition(row, k)
    nearest_neighbors = list()
    i = 0
    while len(nearest_neighbors) != k:
        if neighbors_indexes[i] == rowIndex:
            i += 1
            continue
        nearest_neighbors.append(neighbors_indexes[i])
        i += 1
    return nearest_neighbors


def find_k_nearest_neighbors_for_matrix(matrix, k):
    nearest_neighbors = list()
    for ii, i in enumerate(matrix):
        nearest_neighbors.append(find_k_nearest_neighbors(i, k, ii))
    return nearest_neighbors


def get_total_edges_of_matrix(matrix):
    total = 0
    for ii, i in enumerate(matrix):
        for j in range(ii, len(matrix)):
            if matrix[ii][j] == 1:
                total += 1
    return total


def basic_analysis_of_matrix(matrix):
    nodes_count = len((matrix))
    edges_count = get_total_edges_of_matrix(matrix)

    print("Nodes count: ", nodes_count)
    print("Edges count: ", edges_count)
    print("Avg degree : ", edges_count / nodes_count)


def NN_map_to_incidence_matrix(NN):
    incidence_matrix = np.zeros((len(NN), len(NN)), dtype=int)
    for ii, i in enumerate(NN):
        for j in i:
            incidence_matrix[ii][j] = 1
            incidence_matrix[j][ii] = 1
    return incidence_matrix


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


def k_means(dataset, n_clusters, indexes_to_ignore):
    indexes_to_read = min(indexes_to_ignore)
    only_numeric_data = [np.array(row)[0:indexes_to_read] for row in np.array(dataset)[0:len(dataset)]]
    from_string_to_numbers = [np.array(d, dtype=float) for d in only_numeric_data]
    kmeans5 = KMeans(n_clusters=n_clusters)
    y_kmeans5 = kmeans5.fit_predict(from_string_to_numbers)
    print("clusters: \n", kmeans5.cluster_centers_)
    plt.scatter([np.array(d, dtype=float)[0] for d in from_string_to_numbers],
                [np.array(d, dtype=float)[1] for d in from_string_to_numbers], c=y_kmeans5, cmap='rainbow')


def girvan_newman(graph):
    pos = nx.spring_layout(graph)
    comms = community.girvan_newman(graph)
    nodes_of_comms = next(comms)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_comms[0], node_color='r', alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_comms[1], node_color='b', alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_comms[2], node_color='g', alpha=0.8)
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), alpha=0.5, )
    nx.draw_networkx_labels(graph, pos)


def calculate_eff_of_node(nodeIndex, nodeNeighbors, dataset, classIndex):
    sum = 0.0
    for i in nodeNeighbors:
        if dataset[i][classIndex] == dataset[nodeIndex][classIndex]:
            sum += 1.0
    return sum / len(nodeNeighbors)


def calculate_efficiency_of_network_conversion(NN, dataset, classIndex):
    sum_of_weights = 0.0
    for ii, i in enumerate(NN):
        sum_of_weights += calculate_eff_of_node(ii, i, dataset, classIndex)
    return sum_of_weights / len(dataset)


def run_conversion(fileName, hasHeader, k, epsilon):
    start1 = time.time()
    dataset = read_data_set(fileName, ",", hasHeader)
    matrix = pairwise.euclidean_distances(np.array(dataset))
    EKNN = knn_and_epsilon_combination_method(matrix, k, epsilon)
    # export_edges(EKNN,"edges_export.csv")
    export_nodes_with_attributes(dataset, 7, "nodes_export_by_color.csv")
    end1 = time.time()
    g =create_net_graph(EKNN)
    hist_data = nx.degree_histogram(g)
    plt.hist(hist_data,  facecolor='blue', alpha=0.5)
    plt.show()
    print(end1 - start1)


print("1")
# dataset = read_data_set("small_iris.csv", ";", True)
# dataset = read_data_set("processed_diamonds_2.csv", ",", True)
# export(dataset)
# dataset = read_data_set("iris_with_dots.csv", ";", True)
# matrix = create_matrix_by_euclid_dist(dataset, [4])
# start1 = time.time()
# matrix = pairwise.euclidean_distances(np.array(dataset))
# end1 = time.time()
# print(end1 - start1)
# print("2")
# start = time.time()
# matrix = create_matrix_by_euclid_dist(dataset, [])
# end = time.time()
# print(end - start)
# print("3")
# start1 = time.time()
# EKNN = knn_and_epsilon_combination_method(matrix, 4, 0.4)
# end1 = time.time()
# print(end1 - start1)
#
# print("4")
# export_edges(EKNN)
# export_nodes_with_attributes(dataset, 8)
# start1 = time.time()
# ENN = find_neighbours_by_epsilon_for_matrix(matrix, 0.4)
# end1 = time.time()
# print(end1 - start1)
# incidence_matrix = NN_map_to_incidence_matrix(NN)
# basic_analysis_of_matrix(incidence_matrix)
#
# # k-means vs community detection
# k_means(dataset, 3, [4])
# graph = incidence_matrix_to_graph(incidence_matrix)
#
# # pos = nx.spring_layout(graph, k=0.3 * 1 / np.sqrt(len(graph.nodes())), iterations=20)
# # plt.figure(3, figsize=(30, 30))
# # nx.draw(graph, pos=pos)
# # nx.draw_networkx_labels(graph, pos=pos)
# girvan_newman(graph)
# eff = calculate_efficiency_of_network_conversion(NN, dataset, 4)
# find_neighbours_by_epsilon(matrix[0], 0.4, 0)
# print("DONE")
run_conversion("processed_diamonds_2.csv", True, 4, 0.4)
plt.show()
