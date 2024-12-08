"""From Hugging Face, downloads dataset, organizes them by number of data inputs, 
counts the number of data per each class, and tells what percentage that class makes up of the whole dataset."""

from datasets import load_dataset
from collections import Counter
import numpy as np

# Load a dataset from Hugging Face (HF_username/dataset_name)
dataset = load_dataset('chrismontes/DogData')

# Assuming the 'label' field contains the class information
labels = dataset['train']['label']

# Count the number of instances per class
counter = Counter(labels)

# Calculate the percentage of the whole dataset for each class
percentages = {k: (v / len(labels)) * 100 for k, v in counter.items()}

# Sort classes by number of instances
sorted_classes = sorted(counter.items(), key=lambda x: x[1], reverse=True)

# Print the sorted classes with their counts and percentages
for label, count in sorted_classes:
    print(f'Class: {label}, Count: {count}, Percentage: {percentages[label]:.2f}%')

# Print the number of unique classes
print(f'\nNumber of unique classes: {len(counter)}')
