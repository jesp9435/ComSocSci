import numpy as np
import matplotlib.pyplot as plt

N = 10000
mu = 0
sigma = 4
X = np.random.standard_normal(N) * sigma + mu
plt.hist(X, bins=50, density=True, color='skyblue', alpha=0.7)
plt.title('Histogram of Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

cumulative_averages = np.zeros(len(X))
cumulative_avg_sum = 0
cumulative_standard_error = np.zeros(len(X))
cumulative_std_dev = 0

for i in range(len(X)):
    cumulative_avg_sum += X[i]
    cumulative_averages[i] = cumulative_avg_sum / (i + 1)
print(cumulative_averages)

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

# Show the plot
plt.grid(True)
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