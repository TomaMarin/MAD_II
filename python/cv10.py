import networkx as nx
import random
import matplotlib.pyplot as plt
from networkx.generators.random_graphs import _random_subset
import numpy as np
import copy


def barabasi_albert_graph(n, m, seed=None):
    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))
    if seed is not None:
        random.seed(seed)

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(m)
    G.name = "barabasi_albert_graph(%s,%s)" % (n, m)
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip(*m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend(*m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        targets = _random_subset(repeated_nodes, m)
        source += 1
    return G


def erdos_renyi_graph(n, m):
    G = nx.gnm_random_graph(n, m)
    return G


def random_node_sampling(G, p):
    nodes_of_graph = copy.deepcopy(np.array(G.nodes))
    edges_of_graph = copy.deepcopy(np.array(G.edges))
    list_of_nodes = list()
    list_of_edges = list()
    while len(list_of_nodes) < p * len(nodes_of_graph):
        random_node = nodes_of_graph[random.randint(0, len(nodes_of_graph) - 1)]
        if not np.isin(random_node, list_of_nodes):
            list_of_nodes.append(random_node)
    for i in edges_of_graph:
        if np.isin(i[0], list_of_nodes) and np.isin(i[1], list_of_nodes):
            list_of_edges.append(i)
    sample_graph = nx.empty_graph(len(list_of_nodes))
    sample_graph.name = "sample graph"
    sample_graph.add_edges_from(list_of_edges)
    return sample_graph


def random_degree_node_sampling(G, p):
    nodes_of_graph = copy.deepcopy(np.array(G.nodes))
    edges_of_graph = copy.deepcopy(np.array(G.edges))
    nodes_and_degrees = np.array(copy.deepcopy((G.degree(G.nodes))))
    nodes_and_probs = list()
    for i in nodes_and_degrees:
        nodes_and_probs.append((i[0], (i[1] / nodes_and_degrees.max(0)[1])))
    list_of_nodes = list()
    list_of_edges = list()
    while len(list_of_nodes) < p * len(nodes_of_graph):
        random_node = random.choice([i for i in nodes_and_probs if i[1] > random.random()])[0]
        if not np.isin(random_node, list_of_nodes):
            list_of_nodes.append(random_node)
    for i in edges_of_graph:
        if np.isin(i[0], list_of_nodes) and np.isin(i[1], list_of_nodes):
            list_of_edges.append(i)
    sample_graph = nx.empty_graph(len(list_of_nodes))
    sample_graph.name = "sample graph"
    sample_graph.add_edges_from(list_of_edges)
    return sample_graph


BAG = nx.barabasi_albert_graph(1000, 2)
BAG_node = random_node_sampling(BAG, 0.15)
BAG_edge = random_degree_node_sampling(BAG, 0.15)

ERG = erdos_renyi_graph(1000,2)
ERG_node = random_node_sampling(ERG, 0.15)
ERG_edge = random_degree_node_sampling(ERG, 0.15)
# FOR BAG
# plt.subplot(2, 1, 1)
# nx.draw(BAG_node, with_labels=True)
# # plt.show()
# # nx.draw(G, with_labels=True)
# plt.subplot(2, 1, 2)
# nx.draw(BAG_edge, with_labels=True)
# plt.show()

# FOR ERG
plt.subplot(2, 1, 1)
nx.draw(ERG_node, with_labels=True)
# plt.show()
# nx.draw(G, with_labels=True)
plt.subplot(2, 1, 2)
nx.draw(ERG_edge, with_labels=True)
plt.show()
