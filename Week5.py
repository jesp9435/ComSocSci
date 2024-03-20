import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1: Heavy tailed distributions 
# Exercise 1: Law of large numbers.

# 1) 
'''Sample N=10,000 data points from a Gaussian Distribution with parameters μ=0 and σ=4, 
using the np.random.standard_normal() function. Store your data in a numpy array X'''
N = 10000
mu = 0
sigma = 4
X = np.random.standard_normal(N) * sigma + mu

# 2) 
'''Plot the distribution of the data in X.'''
plt.hist(X, bins=50, density=True, color='skyblue', alpha=0.7)
plt.title('Histogram of Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 3)
'''Compute the cumulative average of X (you achieve this by computing average({X[0],...,X[i−1]})
for each index i∈[1,...,N+1] ). Store the result in an array.'''
cumulative_averages = np.zeros(len(X))
cumulative_avg_sum = 0
for i in range(len(X)):
    cumulative_avg_sum += X[i]
    cumulative_averages[i] = cumulative_avg_sum / (i + 1)
print(cumulative_averages)

# 4)
'''In a similar way, compute the cumulative standard error of X. Note: the standard error of a sample is defined as σ_M=σ/(√n), 
where σ is the sample standard deviation and n is the sample size. Store the result in an array.'''
cumulative_std_err = np.zeros(N)
for i in range(1, N + 1):
    cumulative_std_err[i - 1] = np.std(X[:i]) / np.sqrt(i)

# 5)
'''Compute the values of the distribution mean and median using the formulas
you can find on the Wikipedia page of the Gaussian Distribution'''
mean = np.mean(X)
median = np.median(X)

# 6)
'''Create a figure.
- Plot the cumulative average computed in point 3. as a line plot (where the x-axis represent the size of the sample considered, 
and the y-axis is the average).
- Add errorbars to each point in the graph with width equal to the standard error of the mean (the one you computed in point 4).
- Add a horizontal line corresponding to the distribution mean (the one you found in point 5).'''
sample_sizes = np.arange(1, len(X) + 1)
plt.figure(figsize=(10, 6))
plt.errorbar(sample_sizes, cumulative_averages, yerr=cumulative_std_err, fmt='-o', 
            color='blue', ecolor='red', capsize=5, label='Cumulative Average with Error Bars')

plt.axhline(y=mean, color='red', linestyle='--', label='Distribution Mean')
plt.xlabel('Sample Size')
plt.ylabel('Cumulative Average')
plt.title('Cumulative Average with Error Bars and Distribution Mean')
plt.legend()
plt.xscale("log")
plt.grid(True)
plt.show()

# 7)
'''Compute the cumulative median of X (you achieve this by computing median({X[0],...,X[i−1]}) for each index i∈[1,...,N+1]). 
Store the result in an array.'''
cumulative_median = np.zeros(N)
for i in range(1, N + 1):
    cumulative_median[i - 1] = np.median(X[:i])

# 8)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, N + 1), cumulative_median, label='Cumulative Median', color='blue')
plt.axhline(y=median, color='red', linestyle='--', label='Distribution Median')

plt.show()

'''
# Compute mean and median of the distribution
distribution_mean = np.mean(X)
distribution_median = np.median(X)

# Plot the cumulative average and add error bars
plt.figure(figsize=(10, 6))
plt.errorbar(np.arange(1, N + 1), cumulative_avg, yerr=cumulative_std_err, fmt='-o', 
            color='green', ecolor='red', capsize=5, label='Cumulative Average with Error Bars')
plt.axhline(y=distribution_mean, color='black', linestyle='--', label='Distribution Mean')
plt.title('Cumulative Average with Error Bars and Distribution Mean')
plt.xlabel('Sample Size')
plt.ylabel('Cumulative Average')
plt.legend()
plt.grid(True)
plt.show()
'''
# 9)
xm = 1
alpha = 0.5
pareto_samples = np.random.pareto(alpha, N)

# 10)
plt.hist(pareto_samples, bins=10, density=True, color='skyblue')
plt.title('Pareto Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
count, bins, ignored = plt.hist(pareto_samples, 14, density = True) 
plt.show()

cumulative_averages = np.zeros(len(pareto_samples))
cumulative_avg_sum = 0
cumulative_standard_error = np.zeros(len(pareto_samples))
cumulative_std_dev = 0

for i in range(len(pareto_samples)):
    cumulative_avg_sum += pareto_samples[i]
    cumulative_averages[i] = cumulative_avg_sum / (i + 1)


######################################################################################
# 11)
# Load the CSV file into a DataFrame
df = pd.read_csv('papers.csv')

# Access the third column (number of citations)
citations = df.iloc[:, 2]

# Compute mean and median of the entire population
mean_citations = citations.mean()
median_citations = citations.median()

print("Mean number of citations:", mean_citations)
print("Median number of citations:", median_citations)

random_sample = citations.sample(n=10000, random_state=1)
random_sample_mean = random_sample.mean()
random_sample_median = random_sample.median()
print("\nMean number of citations in sample: ", random_sample_mean)
print("\nMedian number of citations in sample: ", random_sample_median)

plt.figure(figsize=(8, 6))
plt.hist(random_sample, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Number of Citations in Random Sample')
plt.xlabel('Number of Citations')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

sample_cumulative_avg = []
count = 0
for i, value in enumerate(random_sample, start=1):
    count += value
    average = count / i
    sample_cumulative_avg.append(average)
print(sample_cumulative_avg)

cumulative_std_error = []
running_sum_squares = 0

for i, value in enumerate(random_sample, start=1):
    running_sum_squares += value ** 2
    variance = running_sum_squares / i - (np.mean(random_sample[:i])) ** 2
    std_error = np.sqrt(variance / i)
    cumulative_std_error.append(std_error)

# Display the cumulative standard error array
print(cumulative_std_error)

# 13

# Part 2:
#What's the problem with random networks as a model for real-world networks according to the argument in section 3.5 (near the end)?
#In the book they compared a random network model to try and fit it on real data.
# The random network model underestimated the size and the frequency of the high degree nodes, as well as the number of low degree nodes. 
# The random network model predicts a larger number of nodes in the vicinity of ‹k› than seen in real networks.

#List the four regimes that characterize random networks as a function of ⟨k⟩.
#Subcritical regime
#Supercritical regime
#Connected regime
#Critical point

#According to the book, why is it a problem for random networks
#(in terms of being a model for real-world networks that the degree-dependent clustering C(k) decreases as a function of k in real-world networks?
# Random network model does not capture the clustering of real networks. 
# Instead real networks have a much higher clustering coefficient than expected for a random network of similar N and L.
# In a random network the local clustering coefficient is independent of the node’s degree and ‹C› depends on the system size as 1/N. 
# In contrast, measurements indicate that for real networks C(k) decreases with the node degrees and is largely independent of the system size 
# Taken together, it appears that the small world phenomena is the only property reasonably explained by the random network model. 
# All other network characteristics, from the degree distribution to the clustering coefficient, are significantly different in real networks. 

# Part 3:
import csv
import networkx as nx
import netwulf as nw
import numpy as np
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

#RG = nx.gnp_random_graph(amount_of_edges, probability, seed=1000, directed=False)



def generate_random_network(node_count, p):
    G = nx.Graph()
    
    G.add_nodes_from(range(node_count))
    for i in range(node_count):
        for j in range(i + 1, node_count):  
            if np.random.uniform(0, 1) < p:
                G.add_edge(i, j)
    
    return G

random_network = generate_random_network(amount_of_nodes,probability)
nx.draw(random_network)