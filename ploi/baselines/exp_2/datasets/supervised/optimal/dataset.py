import torch

from pathlib import Path
import pymimir as mm
from torch.functional import Tensor
from torch.utils.data.dataset import Dataset
from typing import Dict, List, Tuple

def _split(tokens: list) -> list:
    return [token.split(' ') for token in tokens]


def _read_between(index: int, start_line: str, end_line: str, file: list) -> list:
    index += 1
    lines = []
    if file[index] != start_line: raise Exception(start_line)
    while True:
        index += 1
        if file[index] == end_line: break
        else: lines.append(file[index])
    return index, _split(lines)


def _read_state(index: int, file: list) -> list:
    index += 1
    lines = []
    line = file[index]
    if line != "BEGIN_STATE": raise Exception("BEGIN_STATE")
    while True:
        index += 1
        line = file[index]
        if line == "END_STATE": break
        else: lines.append(line)
    return index, _split(lines)


def _read_labeled_state(index: int, file: list) -> list:
    index += 1
    lines = []
    line = file[index]
    if line != "BEGIN_LABELED_STATE":
        raise Exception("BEGIN_LABELED_STATE")
    index += 1
    lines.append(file[index])
    index, state = _read_state(index, file)
    lines.append(state)
    index += 1
    line = file[index]
    if line != "END_LABELED_STATE":
        Exception("END_LABELED_STATE")
    return (index, lines)


def _read_labeled_states(index: int, file: list) -> list:
    index += 1
    transitions = []
    line = file[index]
    if line != "BEGIN_STATE_LIST":
        raise Exception("BEGIN_STATE_LIST")
    while file[index + 1] == "BEGIN_LABELED_STATE":
        index, transition = _read_labeled_state(index, file)
        transitions.append(transition)
    index += 1
    line = file[index]
    if line != "END_STATE_LIST":
        raise Exception("END_STATE_LIST")
    return index, transitions


def _decode_predicate(objs_map: dict, preds_map: dict, encoded_predicate: list) -> tuple:
    predicate = preds_map[encoded_predicate[0]]
    arguments = tuple([objs_map[index] for index in encoded_predicate[1:]])
    return (predicate, arguments)


def _decode_predicates(objs_map: dict, preds_map: dict, encoded_predicates: list) -> list:
    return [_decode_predicate(objs_map, preds_map, encoded_predicate) for encoded_predicate in encoded_predicates]


def _intify_predicate(encoded_predicate: list) -> tuple:
    predicate = int(encoded_predicate[0])
    arguments = [int(index) for index in encoded_predicate[1:]]
    return (predicate, arguments)


def _intify_predicates(encoded_predicates: list) -> list:
    return [_intify_predicate(encoded_predicate) for encoded_predicate in encoded_predicates]


def _load_file(file: Path, decode: bool):
    with file.open('r') as fs: lines = [line.strip() for line in fs.readlines()]
    index = -1
    index, objs_map = _read_between(index, "BEGIN_OBJECTS", "END_OBJECTS", lines)
    index, preds_map = _read_between(index, "BEGIN_PREDICATES", "END_PREDICATES", lines)
    index, facts_encoded = _read_between(index, "BEGIN_FACT_LIST", "END_FACT_LIST", lines)
    index, goals_encoded = _read_between(index, "BEGIN_GOAL_LIST", "END_GOAL_LIST", lines)
    index, states_encoded = _read_labeled_states(index, lines)
    objs_map = dict(objs_map)
    preds_map = dict(preds_map)
    if decode:
        objs = list(objs_map.values())
        preds = list(preds_map.values())
        facts = _decode_predicates(objs_map, preds_map, facts_encoded)
        goals = _decode_predicates(objs_map, preds_map, goals_encoded)
        states = [(c, _decode_predicates(objs_map, preds_map, state)) for c, state in states_encoded]
    else:
        objs = [int(o) for o in objs_map.keys()]
        preds = [int(p) for p in preds_map.keys()]
        facts = _intify_predicates(facts_encoded)
        goals = _intify_predicates(goals_encoded)
        states = [(c, _intify_predicates(state)) for c, state in states_encoded]
    return {
        'objs': objs,
        'preds': preds,
        'facts': facts,
        'goals': goals,
        'states': states
    }


def _arity_of(predicate, facts, goals, states):
    def find_arity(preds):
        for (other_predicate, arguments) in preds:
            if predicate == other_predicate:
                return len(arguments)
    arity = find_arity(facts)
    if arity != None: return arity
    arity = find_arity(goals)
    if arity != None: return arity
    for (_, state) in states:
        arity = find_arity(state)
        if arity != None: return arity
    return 0


def _pack_by_predicate(predicates, to_tensor: bool):
    packed = {}
    for predicate, arguments in predicates:
        if predicate not in packed: packed[predicate] = []
        packed[predicate].append(arguments)
    if to_tensor:
        for predicate in packed.keys():
            packed[predicate] = torch.tensor(packed[predicate])
    return packed


class ValueDataset(Dataset):
    """State value dataset."""

    def __init__(self, file: Path, min_cost: float = None, max_cost: float = None, decode: bool = False):
        """
        directory (Path): Path to directory of *.txt files with state transitions.
        """
        self._decoded = decode
        data = _load_file(file, decode)

        initial_preds = [(predicate, _arity_of(predicate, data['facts'], data['goals'], data['states'])) for predicate in data['preds']]
        goal_predicate_offset = '_G' if decode else len(initial_preds)
        goal_preds = [(predicate + goal_predicate_offset, arity) for predicate, arity in initial_preds]
        preds = initial_preds + goal_preds

        self.file = file
        self.objects = data['objs']
        self.facts = data['facts']
        self.goals = [(predicate + goal_predicate_offset, arguments) for predicate, arguments in data['goals']]
        if min_cost is not None and max_cost is not None:
            self.states = [state for state in data['states'] if (float(state[0]) >= min_cost) and (float(state[0]) <= max_cost)]
        elif min_cost is not None:
            self.states = [state for state in data['states'] if float(state[0]) >= min_cost]
        elif max_cost is not None:
            self.states = [state for state in data['states'] if float(state[0]) <= max_cost]
        else:
            self.states = data['states']
        self.predicates = preds
        self.predicates.sort()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        (cost, state) = self.states[idx]
        if self._decoded:
            input = _pack_by_predicate(self.facts + self.goals + state, False)
            target = float(cost)
        else:
            input = _pack_by_predicate(self.facts + self.goals + state, True)
            target = torch.tensor([float(cost)])
        return (input, target)

class ValueDataset_mimir(Dataset):
    """State value dataset."""

    def __init__(self, state_space: mm.StateSpace, min_cost: float = None, max_cost: float = None, decode: bool = False):
        #self._decoded = decode
        #data = _load_file(file, decode)
        data = {}
        '''
        data['objs'] = state_space.get_problem().get_objects() 
        data['facts'] = state_space.get_problem().get_facts()
        data['goals'] = state_space.get_problem().get_goals()

        initial_preds = [(predicate, _arity_of(predicate, data['facts'], data['goals'], data['states'])) for predicate in data['preds']]
        goal_predicate_offset = '_G' if decode else len(initial_preds)
        goal_preds = [(predicate + goal_predicate_offset, arity) for predicate, arity in initial_preds]
        preds = initial_preds + goal_preds

        #self.file = file
        self.objects = data['objs']
        self.facts = data['facts']
        self.goals = [(predicate + goal_predicate_offset, arguments) for predicate, arguments in data['goals']]
        if min_cost is not None and max_cost is not None:
            self.states = [state for state in data['states'] if (float(state[0]) >= min_cost) and (float(state[0]) <= max_cost)]
        elif min_cost is not None:
            self.states = [state for state in data['states'] if float(state[0]) >= min_cost]
        elif max_cost is not None:
            self.states = [state for state in data['states'] if float(state[0]) <= max_cost]
        else:
            self.states = data['states']
        self.predicates = preds
        self.predicates.sort()
        '''
        #self.states = state_space.get_problem().get_states()
        self.states = state_space.get_vertices()
        self.targets = state_space.get_goal_distances()
        self.goals = state_space.get_goals()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        #(cost, state) = self.states[idx]
        (cost, state) = self.targets[idx], self.states[idx].get_state()
        if self._decoded:
            input = _pack_by_predicate(self.facts + self.goals + state, False)
            target = float(cost)
        else:
            input = _pack_by_predicate(self.facts + self.goals + state, True)
            target = torch.tensor([float(cost)])
        return (input, target)

def create_dataset_from_state_spaces(state_spaces, max_samples_per_value):
    datasets = []
    for state_space in state_spaces:
        datasets.append((LimitedDataset(ValueDataset_mimir(state_space)), max_samples_per_value))

    return ExtendedDataset(datasets, 1)

class LimitedDataset(Dataset):
    def __init__(self, dataset, max_samples_per_value) -> None:
        super().__init__()
        samples_by_value = {}
        for input, target in dataset:
            key = int(target)
            if key not in samples_by_value:
                samples_by_value[key] = []
            value_samples = samples_by_value[key]
            if len(value_samples) < max_samples_per_value:
                value_samples.append((input, target))
        self.samples = [sample for samples in samples_by_value.values() for sample in samples]
        self.predicates = dataset.predicates

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ExtendedDataset(Dataset):
    def __init__(self, datasets, repeat = 1):
        self._datasets = datasets
        self._repeat = repeat

    def __getitem__(self, index):
        for dataset in self._datasets:
            if index < len(dataset) * self._repeat:
                return dataset[index % len(dataset)]
            else:
                index -= len(dataset) * self._repeat
        raise IndexError()

    def __len__(self):
        return sum(len(d) for d in self._datasets) * self._repeat


def load_dataset(path: Path, max_samples_per_value: int):
    datasets = [LimitedDataset(ValueDataset(d, max_cost=None, decode=False), max_samples_per_value) for d in path.glob('*states.txt')]
    predicates = datasets[0].predicates
    return (ExtendedDataset(datasets, 1), predicates)

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