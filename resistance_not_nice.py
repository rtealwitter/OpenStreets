import networkx as nx
import numpy.linalg as linalg
import numpy as np

DIM = 10
seed = 15 # True
seed = 24 # False

graph = nx.erdos_renyi_graph(DIM, .5, seed=seed)
Laplacian = nx.laplacian_matrix(graph).todense()
print(Laplacian)

def e_i(i):
    e = np.zeros((DIM, 1))
    e[i] = 1
    return e

def resistance(graph, node1, node2):
    Laplacian = nx.laplacian_matrix(graph).todense()
    Lpinv = linalg.pinv(Laplacian)
    characteristic = e_i(node1) - e_i(node2)
    return characteristic.T @ Lpinv @ characteristic

assert nx.is_connected(graph)

s, t = 0,1
i,j = 2,3
k, l = 4,5
print(Laplacian[s,t])
print(Laplacian[i,j])
print(Laplacian[k,l])
R = resistance(graph, s, t)
graph.add_edge(i,j)
R_ij = resistance(graph, s, t)
graph.add_edge(k,l)
R_ij_kl = resistance(graph, s, t)
graph.remove_edge(i,j)
R_kl = resistance(graph, s, t)

print(R_ij - R >= R_ij_kl - R_kl)
print(R_ij - R)
print(R_ij_kl - R_kl)
