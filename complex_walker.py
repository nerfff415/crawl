import random
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def get_subsimplices(simplicial_complex):
    subsimplices = set()
    for simplex in simplicial_complex:
        for i in range(len(simplex)):
            for sub_simplex in itertools.combinations(simplex, i+1):
                subsimplices.add(tuple(sorted(sub_simplex)))
    return list(subsimplices)

def get_faces(K, d2, d1):
    faces = {}
    for simplex in K:
        if len(simplex) == d2 + 1:
            faces[simplex] = []
            for subset in itertools.combinations(simplex, d1 + 1):
                faces[simplex].append(tuple(sorted(subset)))
    return faces

def get_cofaces(K, d1, d2):
    cofaces = {}
    for simplex in K:
        if len(simplex) == d1 + 1:
            cofaces[simplex] = []
            for other_simplex in K:
                if len(other_simplex) == d2 + 1 and set(simplex).issubset(set(other_simplex)):
                    cofaces[simplex].append(other_simplex)
    return cofaces

def get_dual_graph(K, d1, d2):
    G = nx.Graph()
    for simplex in K:
        if len(simplex) == d1 + 1:
            G.add_node(simplex)
    for simplex1, simplex2 in itertools.combinations(K, 2):
        if len(simplex1) == d1 + 1 and len(simplex2) == d1 + 1:
            if d1 <= d2:
                common_cofaces = set(get_cofaces(K, d1, d2)[simplex1]).intersection(set(get_cofaces(K, d1, d2)[simplex2]))
                if common_cofaces:
                    G.add_edge(simplex1, simplex2)
            else:
                common_faces = set(get_faces(K, d1, d2)[simplex1]).intersection(set(get_faces(K, d1, d2)[simplex2]))
                if common_faces:
                    G.add_edge(simplex1, simplex2)
    return G

def random_walk_on_simplicial_complex(simplicial_complex, d_1, d_2, l, m=0, n=0, return_prob=False):
    subsimplices = get_subsimplices(simplicial_complex)
    G = get_dual_graph(subsimplices, d_1, d_2)
    start_node = random.choice([node for node in subsimplices if len(node) == d_1 + 1])
    walk = [start_node]
    prob = 1.0
    for i in range(l - 1):
        current_node = walk[-1]
        neighbors = list(G.neighbors(current_node))
        if len(neighbors) == 0:
            break
        if random.random() < m:
            continue
        if len(walk) > 1 and random.random() < prob * (1-l):
            walk.pop()
            prob /= len(list(G.neighbors(walk[-1])))
            continue
        next_node = random.choice(neighbors)
        if n > 0 and len(walk) > 1 and next_node == walk[-2] and random.random() < n:
            next_node = walk[-1]
        prob /= len(neighbors)
        walk.append(next_node)
    if return_prob:
        return walk,round(prob, 8)
    else:
        return walk

def clique_complex(G):
    cliques = list(nx.find_cliques(G))
    simplices = set()
    for c in cliques:
        for i in range(1, len(c) + 1):
            for simplex in itertools.combinations(c, i):
                simplices.add(simplex)
    return sorted(list(simplices), key=len)

G = nx.karate_club_graph()
simplicial_complex = clique_complex(G)
d_1 = 2
d_2 = 1
l = 10
for i in range(10):
  walk = random_walk_on_simplicial_complex(simplicial_complex, d_1, d_2, l)
  print(walk)