import torch

from torch import tensor
from typing import Dict, List, Tuple


def collate_drop_label(batch: List[Tuple[Dict[int, tensor], int]]):
    """
    Input: [(state, cost)]
    Output: (states, sizes)
    """
    input = {}
    sizes = []
    offset = 0
    for state, _ in batch:
        max_size = 0
        for predicate, values in state.items():
            if values.nelement() > 0:
                max_size = max(max_size, int(torch.max(values)) + 1)
            if predicate not in input: input[predicate] = []
            input[predicate].append(values + offset)
        sizes.append(max_size)
        offset += max_size
    for predicate in input.keys():
        input[predicate] = torch.cat(input[predicate]).view(-1)
    return (input, sizes)


def collate_no_label(batch, device):
    """
    Input: [state]
    Output: (states, sizes)
    """
    input = {}
    sizes = []
    offset = 0
    for state in batch:
        max_size = 0
        for predicate, arguments in state:
            if len(arguments) > 0:
                max_size = max(max_size, max(arguments) + 1)
            if predicate not in input: input[predicate] = []
            input[predicate].append(torch.tensor(arguments) + offset)
        sizes.append(max_size)
        offset += max_size
    for predicate in input.keys():
        input[predicate] = torch.cat(input[predicate]).view(-1).to(device)
    return (input, sizes)