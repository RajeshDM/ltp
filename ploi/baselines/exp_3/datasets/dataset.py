import torch

from datasets.protobuf import LabeledProblem
from pathlib import Path
from random import shuffle
from timeit import default_timer as timer
from termcolor import colored

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

def pack_by_predicate(predicates):
    packed = {}
    for predicate, arguments in predicates:
        if predicate not in packed: packed[predicate] = []
        packed[predicate].append(arguments)
    for predicate, tensors in packed.items():
        packed[predicate] = torch.cat(tensors)
    return packed

def _parse_atoms(atoms, to_predicate):
    #print(f'_parse_atoms: atoms={[(to_predicate[atom.PredicateId], list(atom.ObjectIds)) for atom in atoms]}')
    return [ (to_predicate[atom.PredicateId], torch.tensor(atom.ObjectIds)) for atom in atoms ]

# A parsed state is a dict that maps each predicate symbol to a flat tensor that contains
# the denotation of the predicate; e.g., the following is a parsed state for blocks
#   {'on': tensor([2, 0, 3, 2, 0, 1]), 'handempty': tensor([]), 'ontable': tensor([1]), 'clear': tensor([3]), 'on_goal': tensor([0, 1, 1, 2, 2, 3])}
# In this state, the denotation of 'on' is {(2,0), (3,2), (0,1)}
def _parse_state(state, facts, goals, to_predicate):
    atoms = _parse_atoms(state.Atoms, to_predicate)
    atoms.extend(facts)
    atoms.extend(goals)
    #print(f'_parse_state: atoms={atoms}')
    #print(f'_parse_state: packed={pack_by_predicate(atoms)}')
    return pack_by_predicate(atoms)

def load_problem(file: Path):
    labeled_problem = LabeledProblem()
    labeled_problem.ParseFromString(file.read_bytes())
    return labeled_problem

def _verify_states(labeled_problem, to_predicate):
    # tests whether each non-goal and non-dead end state has a successor with smaller value
    state_map = dict()
    for labeled_state in labeled_problem.LabeledStates:
        label = labeled_state.Label
        state = labeled_state.State
        atoms = tuple(sorted([ (to_predicate[atom.PredicateId], tuple(atom.ObjectIds)) for atom in state.Atoms ]))
        state_map[atoms] = label

    for labeled_state in labeled_problem.LabeledStates:
        label = labeled_state.Label
        state = labeled_state.State
        successors = labeled_state.SuccessorStates
        number_successors = len(successors)
        if label > 0 and label < 2000000000:
            successors_labels = []
            for child in successors:
                atoms = tuple(sorted([ (to_predicate[atom.PredicateId], tuple(atom.ObjectIds)) for atom in child.Atoms ]))
                successors_labels.append(state_map[atoms])
            label_best_child = min(successors_labels)
            if label_best_child >= label:
                print(colored(f'WARNING: state with label {label} has no better successor (best child has label {label_best_child})', 'magenta', attrs = [ 'bold' ]))

def load_file(file: Path, max_samples_per_file: int, verify_states: bool = False):
    start_time = timer()
    print(f'Loading {file} ... ', end='', flush=True)

    # load protobuf structure containing problem with labeled states
    labeled_problem = load_problem(file)
    number_states = len(labeled_problem.LabeledStates)
    print(f'{number_states} state(s) ', end='', flush=True)

    # get predicates, facts and goal_predicates
    to_predicate = dict([(predicate.Id, predicate.Name) for predicate in labeled_problem.Predicates])
    predicates = [(predicate.Name, predicate.Arity) for predicate in labeled_problem.Predicates]
    facts = _parse_atoms(labeled_problem.Facts, to_predicate)
    goal_predicates = [(predicate + '_goal', object_ids) for predicate, object_ids in _parse_atoms(labeled_problem.Goals, to_predicate)]

    # verify states
    if verify_states:
        _verify_states(labeled_problem, to_predicate)

    # random selection of max_samples_per_file (if specified)
    indices_selected_states = list(range(number_states))
    if max_samples_per_file is not None and max_samples_per_file < number_states:
        shuffle(indices_selected_states)
        indices_selected_states = indices_selected_states[:max_samples_per_file]
        print(f'({max_samples_per_file} sampled) ', end='', flush=True)

    # parse selected states and its successors
    num_states = len(indices_selected_states)
    selected_states = [ labeled_problem.LabeledStates[i] for i in indices_selected_states ]
    parsed_states = [ _parse_state(labeled_state.State, facts, goal_predicates, to_predicate) for labeled_state in selected_states ]
    labels_as_tensors = [ torch.tensor([labeled_state.Label]) for labeled_state in selected_states ]
    labeled_states = [ (labels_as_tensors[i], parsed_states[i]) for i in range(num_states) ]
    successor_lists = [ [ _parse_state(successor, facts, goal_predicates, to_predicate) for successor in labeled_state.SuccessorStates ] for labeled_state in selected_states ]
    solvable_labels = [ torch.ones(len(successors)).bool() for successors in successor_lists ]
    labeled_states_with_successors = [ labeled_states[i] + (successor_lists[i],) for i in range(num_states) ]
    elapsed_time = timer() - start_time

    print(f'{elapsed_time:.3f} second(s)')
    return (predicates, labeled_states_with_successors, solvable_labels)

# Returns 1 (resp. 0) iff state for Spanner is solvable (resp. unsolvable)
def _spanner_solvable(state):
    assert type(state) == dict
    assert 'man' in state and 'at' in state and 'link' in state
    assert state['man'].shape == (1,)

    num_loose = 0 if 'loose' not in state else len(state['loose'])
    if num_loose == 0: return True

    carrying = set() if 'carrying' not in state else set(state['carrying'].reshape((-1, 2))[:,1].numpy())
    useable = set() if 'useable' not in state else set(state['useable'].numpy())
    spanner = set() if 'spanner' not in state else set(state['spanner'].numpy())
    #print(f'carrying={carrying}, useable={useable}, spanner={spanner}')

    bob = int(state['man'])
    bob_loc, spanners_loc = None, []
    for p in state['at'].reshape((-1, 2)):
        if bob == p[0]:
            bob_loc = int(p[1])
        else:
            spanners_loc.append(int(p[1]))

    num_unreachable = 0
    if 'link+' in state:
        tc_link = set()
        for p in state['link+'].reshape((-1, 2)):
            tc_link.add((int(p[0]), int(p[1])))
        for loc in spanners_loc:
            if (loc, bob_loc) in tc_link:
                num_unreachable += 1
    else:
        link_map, link_map_inv, rank = dict(), dict(), dict()
        for p in state['link'].reshape((-1, 2)):
            src, dst = int(p[0]), int(p[1])
            link_map[src] = dst
            link_map_inv[dst] = src
        first = (set(link_map.keys()) - set(link_map_inv.keys())).pop()
        while first in link_map:
            rank[first] = len(rank)
            first = link_map[first]
        rank[first] = len(rank)
        bob_rank = rank[bob_loc]
        for loc in spanners_loc:
            if rank[loc] < bob_rank:
                num_unreachable += 1

    num_needed = num_loose - len(carrying & useable)
    num_on_floor = len(spanner - carrying)
    num_available = num_on_floor - num_unreachable
    #print(f'num_needed={num_needed}, num_on_floor={num_on_floor}, num_available={num_available}')
    return num_needed <= num_available

def load_file_spanner(file: Path, max_samples_per_file: int, verify_states: bool = False):
    start_time = timer()
    print(f'Loading {file} ... ', end='', flush=True)

    # load protobuf structure containing problem with labeled states
    labeled_problem = load_problem(file)
    number_states = len(labeled_problem.LabeledStates)
    print(f'{number_states} state(s) ', end='', flush=True)

    # get predicates, facts and goal_predicates
    to_predicate = dict([(predicate.Id, predicate.Name) for predicate in labeled_problem.Predicates])
    predicates = [(predicate.Name, predicate.Arity) for predicate in labeled_problem.Predicates]
    facts = _parse_atoms(labeled_problem.Facts, to_predicate)
    goal_predicates = [(predicate + '_goal', object_ids) for predicate, object_ids in _parse_atoms(labeled_problem.Goals, to_predicate)]

    # verify states
    if verify_states:
        _verify_states(labeled_problem, to_predicate)

    # random selection of max_samples_per_file (if specified)
    indices_selected_states = list(range(number_states))
    shuffle(indices_selected_states)
    if max_samples_per_file is not None and max_samples_per_file < number_states:
        indices_selected_states = indices_selected_states[:max_samples_per_file]
        print(f'({max_samples_per_file} sampled) ', end='', flush=True)

    # parse selected states and its successors
    num_states = len(indices_selected_states)
    selected_states = [ labeled_problem.LabeledStates[i] for i in indices_selected_states ]
    parsed_states = [ _parse_state(labeled_state.State, facts, goal_predicates, to_predicate) for labeled_state in selected_states ]
    labels_as_tensors = [ torch.tensor([labeled_state.Label]) for labeled_state in selected_states ]
    labeled_states = [ (labels_as_tensors[i], parsed_states[i]) for i in range(num_states) ]
    successor_lists = [ [ _parse_state(successor, facts, goal_predicates, to_predicate) for successor in labeled_state.SuccessorStates ] for labeled_state in selected_states ]
    solvable_labels = [ torch.tensor([ _spanner_solvable(successor) for successor in successors ], dtype=torch.bool) for successors in successor_lists ]
    labeled_states_with_successors = [ labeled_states[i] + (successor_lists[i],) for i in range(num_states) ]
    elapsed_time = timer() - start_time

    for i in range(len(solvable_labels)):
        label = int(labeled_states[i][0])
        labels_sum = sum(solvable_labels[i])
        assert labels_sum == 0 or label < 2000000000, f'labels={solvable_labels[i]}, label={label}, state={labeled_states[i][1]}, solvable={_spanner_solvable(labeled_states[i][1])}'
        assert labels_sum > 0 or (len(solvable_labels[i]) == 0 and label == 0) or label > 2000000000, f'labels={solvable_labels[i]}, label={label}, state={labeled_states[i][1]}, solvable={_spanner_solvable(labeled_states[i][1])}'
        #if labels_sum > 0 and labels_sum < len(solvable_labels[i]): print(f'XXXX={solvable_labels[i]}')

    print(f'{elapsed_time:.3f} second(s)')
    return (predicates, labeled_states_with_successors, solvable_labels)

def load_directory(path: Path, max_samples_per_file: int, max_samples: int, filtering_fn = None, load_file_fn = None, verify_states: bool = False):
    if filtering_fn is not None:
        raise NotImplementedError('Filtering of states (see code for additional info)')
        # Filtering function should be passed to load_file() like max_samples_per_file
        # Filtering function can also be applied after states from all files have been processed

    labeled_states = []
    solvable_labels = []
    files = list(path.glob('*.states'))
    print(f'{len(files)} file(s) to load from {path}')
    load_file_fn_aux = load_file if load_file_fn is None else load_file_fn
    for i, file in enumerate(files):
        print(f'({1+i}/{len(files)}) ', end='')
        predicates, states, _solvable_labels = load_file_fn_aux(file, max_samples_per_file, verify_states)
        assert len(states) == len(_solvable_labels)
        assert max_samples_per_file is None or len(states) <= max_samples_per_file
        labeled_states.extend(states)
        solvable_labels.extend(_solvable_labels)
    assert len(labeled_states) == len(solvable_labels)

    if max_samples is not None and max_samples < len(labeled_states):
        indices_selected_states = list(range(len(labeled_states)))
        shuffle(indices_selected_states)
        indices_selected_states = indices_selected_states[:max_samples]
        labeled_states = [ labeled_states[i] for i in indices_selected_states ]
        solvable_labels = [ solvable_labels[i] for i in indices_selected_states ]
    assert len(labeled_states) == len(solvable_labels)

    # print histogram of labels
    histogram = dict()
    for labeled_state in labeled_states:
        label = int(labeled_state[0][0])
        if label not in histogram:
            histogram[label] = 0
        histogram[label] += 1
    print(f'dataset labels: total={len(labeled_states)}, histogram={histogram}')

    predicates_with_goals = predicates + [(predicate + '_goal', arity) for predicate, arity in predicates]
    return (labeled_states, solvable_labels, predicates_with_goals)

