"""
Created on Tue May  2 03:06:04 2023

@author: dell
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data5.csv')
print(data.head())

with open('data5.csv', 'r') as file:
    weights_str = file.readlines()
weights = [float(w.strip()) for w in weights_str]
histogram = [0] * 10  # We'll use 10 bins for simplicity
bin_size = max(weights) / len(histogram)
for w in weights:
    bin_index = int(w / bin_size)
    if bin_index >= len(histogram):
        bin_index = len(histogram) - 1
    histogram[bin_index] += 1
print(histogram)

weights = []
with open('data5.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        weights.append(float(row[0]))

average_weight = np.mean(weights)
lower_limit = 0.8 * average_weight
upper_limit = 1.2 * average_weight

count = sum(1 for w in weights if lower_limit <= w <= upper_limit)

X = count / len(weights)
n, bins, patches = plt.hist(weights, bins=30, alpha=0.9, label='Newborn Weights')

plt.xlabel('Weight (KG)')
plt.ylabel('Count')
plt.title('Distribution of Newborn Weights')
plt.legend(loc='upper right')
plt.text(1, 0.8, f'Average weight: {average_weight:.2f} KG\nX: {X:.2f}', ha='right', va='top', transform=plt.gca().transAxes)

plt.show()
