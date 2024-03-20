'''
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
'''

import networkx as nx
from collections import defaultdict
from itertools import combinations
from bs4 import BeautifulSoup  
import json
import pandas as pd
import concurrent.futures 
import requests
import math
import csv
from collections import Counter
import ast

papers=[]
with open("T_final_papers.csv","r") as f:
    csv_reader=csv.reader(f)
    for row in csv_reader:
        last_column_as_list = ast.literal_eval(row[-1])
        papers.append(row[:-1] + [last_column_as_list])

author_pairs = defaultdict(int)
duplicate_ids = set()

abstracts_df=pd.read_csv("abstract.csv")
papers_df=pd.read_csv("T_final_papers.csv")
file1 = open('weighted_edgelist.txt', mode='a', encoding='utf-8')
def make_pairs(temp_list):
    temp_list=list(temp_list)
    temp_list.sort()
    pairs=list(combinations(temp_list,2))
    return pairs

pairs=[]
checked_pairs=[]
weighted_edgelist=[]
for i in range(len(papers)):
    if(len(papers[i][3])>1):
        for pair in make_pairs(papers[i][3]):
            pairs.append(pair)
while len(pairs)>0:
    count=0
    index=[]
    temp_pair=pairs[0]
    for i in range(len(pairs)):
        if temp_pair==pairs[i]:
            index.append(i)
            count+=1

    for indexes in index:
        pairs.pop(indexes)

    file1.write(temp_pair[0]+","+temp_pair[1]+","+str(count)+"\n")
    weighted_edgelist.append([temp_pair[0],temp_pair[1],count])