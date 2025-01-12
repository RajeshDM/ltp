import torch

from collections import defaultdict
from datasets import load_directory, load_file_spanner
from pathlib import Path
from random import shuffle
from torch.functional import Tensor
from torch.utils.data.dataset import Dataset
from typing import Dict, List, Tuple


def balance_states_by_label(states: List[Tuple[Tensor, dict, List[dict]]], limit: int):
    balanced_states = []
    by_label = defaultdict(list)
    for state in states:
        by_label[state[0]].append(state)
    label_limit = limit // len(by_label.keys())
    for grouped_states in by_label.values():
        shuffle(grouped_states)
        balanced_states.extend(grouped_states[:label_limit])
    for grouped_states in by_label.values():
        remaining = limit - len(balanced_states)
        balanced_states.extend(grouped_states[label_limit:label_limit + remaining])
    return balanced_states


class SupervisedDataset(Dataset):
    def __init__(self, labeled_states: list, successor_labels: list = None):
        self._states = labeled_states
        self._successor_labels = successor_labels
        assert len(self._states) == len(self._successor_labels)

    def __len__(self):
        return len(self._states)

    def __getitem__(self, idx):
        # Successor states are ignored as this is dataset is intended for supervised learning.
        (cost, state, _) = self._states[idx]
        return (state, cost)


def load_dataset(path: Path, max_samples_per_file: int, max_samples: int, verify: bool = False):
    is_spanner = 'spanner' in str(path) and 'spanner-bidirectional' not in str(path)
    load_file_fn = None if not is_spanner else load_file_spanner
    labeled_states, successor_labels, predicates_with_goals = load_directory(path, max_samples_per_file, max_samples, filtering_fn=balance_states_by_label, load_file_fn=load_file_fn, verify_states=verify)
    return (SupervisedDataset(labeled_states, successor_labels), predicates_with_goals)


def collate(batch: List[Tuple[Dict[int, Tensor], int]]):
    """
    Input: [(state, cost)]
    Output: ((states, sizes), costs)
    """
    input = {}
    sizes = []
    offset = 0
    target = []
    for state, cost in batch:
        max_size = 0
        for predicate, values in state.items():
            if values.nelement() > 0:
                max_size = max(max_size, int(torch.max(values)) + 1)
            if predicate not in input: input[predicate] = []
            input[predicate].append(values + offset)
        sizes.append(max_size)
        offset += max_size
        target.append(cost)
    for predicate in input.keys():
        input[predicate] = torch.cat(input[predicate]).view(-1)
    return ((input, sizes), torch.stack(target))
