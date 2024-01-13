import matplotlib.pyplot as plt
import numpy as np

def generate_violinplot(data, labels, title, xlabel, ylabel):
    plt.violinplot(data, showmedians=True)
    plt.title(title)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Example data (replace this with your own data)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(0, 2, 100)

# List of datasets
data = [data1, data2, data3]

# List of labels for each dataset
labels = ['Dataset 1', 'Dataset 2', 'Dataset 3']

# Customize the title, xlabel, and ylabel
title = 'Violin Plot Example'
xlabel = 'Datasets'
ylabel = 'Values'

# Generate and display the violin plot
generate_violinplot(data, labels, title, xlabel, ylabel)
