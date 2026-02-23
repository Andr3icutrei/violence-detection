import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter


def create_balanced_sampler(dataset):
    all_labels = dataset.labels

    label_counts = Counter(all_labels)

    weights = [1.0 / label_counts[label] for label in all_labels]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler