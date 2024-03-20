import csv
import networkx as nx
import netwulf as nw
edges = []
amount_of_edges = 0
with open("weighted_edgelist.txt","r") as f:
    edges = list(tuple(line) for line in csv.reader(f))
for edge in edges:
    amount_of_edges += int(edge[2])

amount_of_nodes = 79264
print(amount_of_edges)

probability = amount_of_edges/((amount_of_edges*(amount_of_edges-1))/2)
print(probability)

k = probability*(amount_of_edges-1.0)
print(k)

RG = nx.gnp_random_graph(amount_of_edges, probability, seed=1000, directed=False)
nw.visualize(RG)