import networkx as nx
import random
import matplotlib.pyplot as plt
# from networkx.generators.random_graphs import _random_subset
import numpy as np
import copy


def random_subset(seq, m, rng):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = np.random.choice(seq)
        targets.add(x)
    return targets


def holme_alg(n, m, p, seed):
    G = nx.empty_graph(m)  # add m initial nodes (m0 in barabasi-speak)
    repeated_nodes = list(G.nodes())  # list of existing nodes to sample from
    # with nodes repeated once for each adjacent edge
    source = m  # next node is m
    while source < n:  # Now add the other n-1 nodes
        possible_targets = random_subset(repeated_nodes, m, seed)
        # do one preferential attachment for new node
        target = possible_targets.pop()
        G.add_edge(source, target)
        repeated_nodes.append(target)  # add one node to list for each new link
        count = 1
        while count < m:  # add m-1 more new links
            if random.random() < p:  # clustering step: add triangle
                neighborhood = [nbr for nbr in G.neighbors(target)
                                if not G.has_edge(source, nbr)
                                and not nbr == source]
                if neighborhood:  # if there is a neighbor without a link
                    nbr = np.random.choice(neighborhood)
                    G.add_edge(source, nbr)  # add triangle
                    repeated_nodes.append(nbr)
                    count = count + 1
                    continue  # go to top of while loop
            # else do preferential attachment step if above fails
            target = possible_targets.pop()
            G.add_edge(source, target)
            repeated_nodes.append(target)
            count = count + 1

        repeated_nodes.extend([source] * m)  # add source node to list m times
        source += 1
    return G

# FIXME check generation of nodes and edges
def bianconi_model(m0, m, t, prob):
    matrix = np.zeros((m0 + t, m0 + t), dtype=np.int8)
    for i in range(m0, m0 + t):
        r_edges = np.random.choice(np.arange(0, i), size=m)
        for j in r_edges:
            matrix[i][j] = 1
            matrix[j][i] = 1
    for i in range(m0, m0 + t):
        r = np.random.choice(np.arange(0, i - 1))
        matrix[i][r] = 1
        matrix[r][i] = 1
        for k in range(i - 1):
            for l in range(i):
                if matrix[i][l] == 1 and matrix[k][l] == 1:
                    x = np.random.random()
                    if x < prob:
                        matrix[i][k] = 1
                        matrix[k][i] = 1
                else:
                    x = np.random.random()
                    if x < 1 - prob:
                        matrix[i][k] = 1
                        matrix[k][i] = 1
    return matrix


HG = holme_alg(50, 2, 0.2, random.Random)
print(np.array(HG.edges).size)
BM = bianconi_model(2,50,2,0.2)
plt.subplot(2, 1, 1)
nx.draw(HG, with_labels=True)
# plt.subplot(2, 1, 2)
plt.show()
