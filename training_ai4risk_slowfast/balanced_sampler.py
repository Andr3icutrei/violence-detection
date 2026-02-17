import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter


def create_balanced_sampler(dataset):
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label.item())

    label_counts = Counter(all_labels)

    weights = []
    for label in all_labels:
        weights.append(1.0 / label_counts[label])

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler