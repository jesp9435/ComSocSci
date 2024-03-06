import numpy as np
import matplotlib.pyplot as plt
import math

X = np.random.standard_normal(size=10000)
plt.hist(X, bins=50, density=True, color='skyblue', alpha=0.7)
plt.title('Histogram of Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

averages = np.zeros(len(X))
cumulative_sum = 0

for i in range(len(X)):
    cumulative_sum += X[i]
    averages[i] = cumulative_sum / (i + 1)

print(averages)