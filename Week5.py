import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1:
# 1)
N = 10000
mu = 0
sigma = 4
X = np.random.standard_normal(N) * sigma + mu

# 2)
plt.hist(X, bins=50, density=True, color='skyblue', alpha=0.7)
plt.title('Histogram of Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 3)
cumulative_averages = np.zeros(len(X))
cumulative_avg_sum = 0
cumulative_standard_error = np.zeros(len(X))
cumulative_std_dev = 0

for i in range(len(X)):
    cumulative_avg_sum += X[i]
    cumulative_averages[i] = cumulative_avg_sum / (i + 1)
print(cumulative_averages)

# 4)
for i in range(len(X)):
    cumulative_std_dev += np.std(X[:i+1])
    cumulative_standard_error[i] = cumulative_std_dev / np.sqrt(i + 1)

print(cumulative_standard_error)

# Compute the cumulative average of X
cumulative_avg = np.cumsum(X) / np.arange(1, N + 1)
print(cumulative_avg)

# Compute the cumulative standard error of X
cumulative_std_err = np.zeros(N)
for i in range(1, N + 1):
    cumulative_std_err[i - 1] = np.std(X[:i]) / np.sqrt(i)
cumulative_error = np.cumsum(cumulative_std_err)
print(cumulative_error)

# 5)
mean = np.mean(X)
median = np.median(X)

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
# Compute the cumulative median of X
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

# Part 2:
#What's the problem with random networks as a model for real-world networks according to the argument in section 3.5 (near the end)?


#List the four regimes that characterize random networks as a function of ⟨k⟩.

#According to the book, why is it a problem for random networks
#(in terms of being a model for real-world networks that the degree-dependent clustering C(k) decreases as a function of k in real-world networks?