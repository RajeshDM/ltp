import torch

from datasets import load_directory, load_file_spanner
from pathlib import Path
from torch.functional import Tensor
from torch.utils.data.dataset import Dataset
from typing import Dict, List, Tuple

class UnsupervisedDataset(Dataset):
    def __init__(self, labeled_states: list, solvable_labels: list = None):
        self._states = labeled_states
        self._solvable_labels = solvable_labels
        assert len(self._states) == len(self._solvable_labels)

    def __len__(self):
        return len(self._states)

    def __getitem__(self, idx):
        (label, state, successor_states) = self._states[idx]
        state_with_successors = [state]
        solvable_labels = torch.tensor([ label < 2000000000 ]) # Use fn for solvability
        if label > 0:
            state_with_successors.extend(successor_states)
            solvable_labels = torch.cat([ solvable_labels, self._solvable_labels[idx] ])
        return (label, state_with_successors, solvable_labels)

def load_dataset(path: Path, max_samples_per_file: int, max_samples: int, verify: bool = False):
    # TODO: Balance states by label
    is_spanner = 'spanner' in str(path) and 'spanner-bidirectional' not in str(path)
    load_file_fn = None if not is_spanner else load_file_spanner
    labeled_states, solvable_labels, predicates_with_goals = load_directory(path, max_samples_per_file, max_samples, load_file_fn=load_file_fn, verify_states=verify)
    return (UnsupervisedDataset(labeled_states, solvable_labels), predicates_with_goals)

# Make a combined representation of the list of states with successors.
# For this, the objects for each state are uniquely identified so that
# no two states share objects
def collate(batch: List[Tuple[int, List[Dict[int, Tensor]]]]):
    """
    Input: [[(label, state_with_successors, solvable_labels)]]
    Output: (labels, (collated_states, object_counts), states_counts)
    """
    #print(f'collate: len(batch)={len(batch)}')
    #print(f'collate: batch[0]={batch[0]}\n')
    collated_states = {}
    object_counts = []
    offset = 0
    for (_, state_with_successors, _solvable_labels) in batch:
        assert len(state_with_successors) == len(_solvable_labels)
        #print(f'collate: len(state_with_successors)={len(state_with_successors)}')
        for state in state_with_successors:
            max_size = 0
            for predicate, labels in state.items():
                #print(f'collate: predicate={predicate}, labels={labels}')
                if labels.nelement() > 0:
                    max_size = max(max_size, int(torch.max(labels)) + 1)
                if predicate not in collated_states: collated_states[predicate] = []
                collated_states[predicate].append(labels + offset)
            object_counts.append(max_size)
            #print(f'collate: input={input}')
            #print(f'collate: object_counts={object_counts}')
            offset += max_size
        #print(f'solvable_labels={_solvable_labels}')
    for predicate in collated_states.keys():
        collated_states[predicate] = torch.cat(collated_states[predicate]).view(-1)
    labels = torch.cat([ label for label, _, _ in batch ])
    solvable_labels = torch.cat([ _solvable_labels for _, _, _solvable_labels in batch ])
    states_counts = torch.tensor([ len(state_with_successors) for _, state_with_successors, _ in batch ], dtype=torch.int32)
    #print(f'collate: labels={labels}, states_counts={states_counts}')
    return (labels, (collated_states, object_counts), solvable_labels, states_counts)

