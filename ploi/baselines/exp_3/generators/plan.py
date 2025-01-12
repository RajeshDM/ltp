from copy import deepcopy as deepcopy
from pathlib import Path
from termcolor import colored
import pytorch_lightning as pl
from torch.functional import Tensor
from typing import Dict, List, Tuple
import torch

from tarski.fstrips.fstrips import AddEffect, DelEffect
from tarski.model import Model as PDDLState
from tarski.syntax.formulas import Atom
from tarski.model import create as make_tarski_state


def _collate(batch: List[Dict[str, Tensor]], device):
    """
    Input: [state]
    Output: (states, sizes)
    """
    input = {}
    sizes = []
    offset = 0
    for state in batch:
        max_size = 0
        for predicate, values in state.items():
            if values.nelement() > 0:
                max_size = max(max_size, int(torch.max(values)) + 1)
            if predicate not in input: input[predicate] = []
            input[predicate].append(values + offset)
        sizes.append(max_size)
        offset += max_size
    for predicate in input.keys():
        input[predicate] = torch.cat(input[predicate]).view(-1).to(device=device, non_blocking=True)
    return (input, sizes)

def load_pddl_problem_with_augmented_states(domain: Path, problem: Path, registry_filename: Path = None, registry_key: str = None, logger = None):
    from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
    from tarski.errors import UndefinedPredicate
    from tarski.io import PDDLReader

    from sys import path as sys_path
    sys_path.append('../DerivedPredicates')
    from augmentation import load_registry, get_registry_record, update_registry_record, construct_augmentation_function_simple

    parser = PDDLReader(raise_on_error=True)
    parser.parse_domain(str(domain))
    problem = parser.parse_instance(str(problem))
    actions = ground_problem_schemas_into_plain_operators(problem)
    predicates = [ predicate for predicate in problem.language.predicates if '=' not in str(predicate.name) ]
    language = problem.language

    # calculate sets of static and dynamic predicates
    all_predicates = set([ predicate.name for predicate in predicates ])
    dynamic_predicates = set()
    for action_name in problem.actions:
        action = problem.actions[action_name]
        for effect in action.effects:
            dynamic_predicates.add(str(effect.atom.predicate.name))
    static_predicates = all_predicates - dynamic_predicates
    if logger: logger.info(f'Predicates: static={static_predicates}, dynamic={dynamic_predicates}')

    # set augmentation function if registry file and proper key found
    augment_fn = None
    if registry_filename is not None:
        # calculate registry_key from domain path if needed
        if not registry_key:
            known_domains = set([ dname.name for dname in Path('../Data/pddl').glob('*') if dname.is_dir() ])
            domain_parts = set(domain.parts)
            intersection = known_domains & domain_parts
            registry_key = intersection.pop() if len(intersection) == 1 else None

        # get augmentation function
        if registry_key:
            if logger: logger.info(f"Using '{registry_key}' as registry key into '{registry_filename}'")
            registry = load_registry(registry_filename)
            registry_record = get_registry_record(registry_key, registry, registry_filename)
            update_registry_record(registry_record, static_predicates, dynamic_predicates)
            augment_fn = construct_augmentation_function_simple(registry_record)

            # extend language with derived and goal predicates
            if augment_fn is not None:
                # setup 'arities' for all predicates in language
                for predicate in language.predicates:
                    if '=' not in str(predicate.name):
                        registry_record['arities'][str(predicate.name)] = predicate.arity

                # get object 'object'
                try:
                    obj = language.get_sort('object')
                except:
                    if logger: logger.fatal(f"domain doesn't define 'object' type")
                    exit(-1)

                # extend language
                for predicate_name in registry_record['static'] | registry_record['dynamic']:
                    arity = registry_record['arities'][predicate_name]
                    args = [ obj for i in range(arity) ]
                    try:
                        language.get_predicate(predicate_name)
                        if predicate_name in registry_record['dynamic']:
                            predicate = language.predicate(predicate_name + '@', *args)
                            if logger: logger.info(f"Extending domain language with predicate '{predicate}'")
                    except UndefinedPredicate:
                        predicate = language.predicate(predicate_name, *args)
                        if logger: logger.info(f"Extending domain language with predicate '{predicate}'")
                    predicates = [ predicate for predicate in problem.language.predicates if '=' not in str(predicate.name) ]
        else:
            if logger: logger.warning(f'Unable to calculate registry key; bypassing registry ...')

    return {
        'actions': actions,
        'initial': problem.init,
        'goal': problem.goal,
        'predicates': predicates,
        'language': language,
        'augment_fn': augment_fn
    }

def load_pddl_problem(domain: Path, problem: Path, logger = None):
    from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
    from tarski.io import PDDLReader
    parser = PDDLReader(raise_on_error=True)
    parser.parse_domain(str(domain))
    problem = parser.parse_instance(str(problem))
    actions = ground_problem_schemas_into_plain_operators(problem)
    predicates = [ predicate for predicate in problem.language.predicates if '=' not in str(predicate.name) ]
    language = problem.language

    return {
        'actions': actions,
        'initial': problem.init,
        'goal': problem.goal,
        'predicates': predicates,
        'language': language,
        'augment_fn' : None
    }

def create_object_encoding(objects):
    object_ids = dict([ (obj.name, i) for i, obj in enumerate(objects) ])
    object_ids_inv = dict([ (i, obj.name) for i, obj in enumerate(objects) ])
    object_ids.update(object_ids_inv)
    return object_ids

def _get_goal_denotation(goals, obj_encoding: Dict[str, int]) -> Dict[str, List[int]]:
    atoms = [ goals ] if isinstance(goals, Atom) else goals.subformulas
    predicate_names = [ atom.predicate.name + '_goal' for atom in atoms ]
    argument_ids = [ [ obj_encoding[obj.name] for obj in atom.subterms ] for atom in atoms]
    denotation = dict()
    for predicate_name, object_ids in zip(predicate_names, argument_ids):
        if predicate_name not in denotation:
            denotation[predicate_name] = []
        denotation[predicate_name].extend(object_ids)
    return denotation

def _encode_atoms(state: PDDLState, object_ids: Dict[str, int], logger = None) -> dict:
    encoded_atoms = {}
    for atom in state.as_atoms():
        if hasattr(atom, 'predicate'):
            predicate_name = atom.predicate.name
            encoded_objects = [ object_ids[obj.name] for obj in atom.subterms ]
            if predicate_name not in encoded_atoms: encoded_atoms[predicate_name] = []
            encoded_atoms[predicate_name].extend(encoded_objects)
    #if logger: logger.debug(f'encoded_atoms={encoded_atoms}')
    return encoded_atoms

def _encode_state(state: PDDLState, goal_denotation, object_ids: Dict[str, int], logger = None) -> dict:
    encoded_state = _encode_atoms(state, object_ids, logger)
    for predicate_name in goal_denotation:
        encoded_state[predicate_name] = goal_denotation[predicate_name]
    #if logger: logger.debug(f'encoded_state={encoded_state}')
    return encoded_state

def _to_tensors(encoded_states: List[Dict[str, List[int]]]):
    for encoded_state in encoded_states:
        for pred_id, obj_ids in encoded_state.items():
            encoded_state[pred_id] = torch.tensor(obj_ids)

def _to_input(states: List[PDDLState], goal_denotation, obj_encoding, augment_fn, language, device, logger = None):
    encoded_states = [ _encode_state(_apply_derived_predicates(state, goal_denotation, obj_encoding, augment_fn, language), goal_denotation, obj_encoding, logger) for state in states ]
    _to_tensors(encoded_states)
    return _collate(encoded_states, device), encoded_states

def _get_applicable_actions(state, actions):
    return [ operator for operator in actions if state[operator.precondition] ]

def _apply_action(state, action):
    state = deepcopy(state)
    delete_atoms = [ effect.atom for effect in action.effects if isinstance(effect, DelEffect) ]
    add_atoms = [ effect.atom for effect in action.effects if isinstance(effect, AddEffect) ]
    for delete_atom in delete_atoms: state.discard(delete_atom.predicate, *delete_atom.subterms)
    for add_atom in add_atoms: state.add(add_atom.predicate, *add_atom.subterms)
    return state

def _get_successor_states(state, actions):
    applicable_actions = _get_applicable_actions(state, actions)
    return [ (action, _apply_action(state, action)) for action in applicable_actions ]

def _map_tarski_state_into_plain_state(tarski_state: PDDLState):
    plain_state = dict(_internal=dict())
    for key in tarski_state.predicate_extensions:
        #plain_state[key[0]] = tarski_state.predicate_extensions[key]
        plain_state[key[0]] = set([ tuple([ str(wref.expr) for wref in d ]) for d in tarski_state.predicate_extensions[key] ])
        plain_state['_internal'][key[0]] = key
    return plain_state

def _augment_tarski_state(tarski_state: PDDLState, augmentation, language):
    for key in augmentation:
        predicate = language.get_predicate(key)
        for d in augmentation[key]:
            #denotation = tuple([ wref.expr for wref in d ])
            #tarski_state.add(predicate, *denotation)
            tarski_state.add(predicate, *d)
    return tarski_state

def _apply_derived_predicates(tarski_state: PDDLState, goal_denotation, obj_encoding, augment_fn, language):
    assert augment_fn is None or language is not None, f'language={language}, augment_fn={augment_fn}'
    if augment_fn is not None:
        # get plain state from PDDL (tarski) state
        plain_state = _map_tarski_state_into_plain_state(tarski_state)

        # augment plain state with goal predicates of type <pred>@
        added_goal_denotations = []
        for predicate_name in goal_denotation:
            assert predicate_name[-5:] == '_goal'
            goal_predicate = language.get_predicate(predicate_name[:-5] + '@')
            denotation = goal_denotation[predicate_name]
            arity = goal_predicate.arity
            plain_state[str(goal_predicate.name)] = set()
            for i in range(0, len(denotation), arity):
                plain_state[str(goal_predicate.name)].add(tuple(map(lambda i: obj_encoding[i], denotation[i:i+arity])))
            assert goal_predicate.name not in plain_state['_internal']
            plain_state['_internal'][goal_predicate.name] = goal_predicate.signature
            added_goal_denotations.append(goal_predicate.name)

        # augment plain state with derived predicates
        augmented_state = augment_fn(plain_state, 'static')
        augmented_state = augment_fn(augmented_state, 'dynamic')

        # calculate augmentation
        keys = [ key for key in augmented_state if key not in plain_state ]
        augmentation = dict([ (key, augmented_state[key]) for key in keys ])
        return _augment_tarski_state(deepcopy(tarski_state), augmentation, language)
    else:
        return tarski_state

def _spanner_solved(state, logger=None):
    # assuming that unsolvable check was previously done, task is solvable if bob is at gate

    # extract information from state
    extensions = state.list_all_extensions()
    at = dict([ (obj1.name, obj2.name) for (obj1, obj2) in extensions[('at', 'object', 'object')] ])

    # get bob location
    assert len(extensions[('man', 'object')]) == 1
    bob = extensions[('man', 'object')][0][0].name
    bob_loc = at[bob]
    return bob_loc == 'gate'

def _spanner_unsolvable(state, logger=None):
    # if bob is at gate, check he has enough useable spanners

    # extract information from state
    extensions = state.list_all_extensions()
    at = dict([ (obj1.name, obj2.name) for (obj1, obj2) in extensions[('at', 'object', 'object')] ])

    # get bob location
    assert len(extensions[('man', 'object')]) == 1
    bob = extensions[('man', 'object')][0][0].name
    bob_loc = at[bob]
    if bob_loc != 'gate': return False

    # get spanner and nut subsets
    carrying = set([ obj2.name for (obj1, obj2) in extensions[('carrying', 'object', 'object')] if obj1.name == bob ])
    useable = set([ obj.name for (obj,) in extensions[('useable', 'object')] ])
    spanner = set([ obj.name for (obj,) in extensions[('spanner', 'object')] ])
    loose_nuts = set([ obj.name for (obj,) in extensions[('loose', 'object')] ])
    carrying_and_useable_spanners = carrying & useable & spanner

    # at this point, task is solvable iff bob is carrying enough useable spanners
    if logger and len(carrying_and_useable_spanners) < len(loose_nuts):
        logger.info(f'unsolvable_state={state}')
        logger.info(f'carrying={carrying}, useable={useable}, spanner={spanner}, loose_nuts={loose_nuts}')
        logger.info(f'carrying_and_useable_spanners={carrying_and_useable_spanners}')
    return len(carrying_and_useable_spanners) < len(loose_nuts)

def policy_search(actions, initial, goals, obj_encoding: Dict[str, int], model: pl.LightningModule, cycles: str = 'avoid', max_state_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None):
    language = None
    augment_fn = None

    device = model.device
    closed_states = set()
    action_trace = []

    # calculate denotation of goal atoms that is equal for every state
    if logger: logger.info(f'goals={goals}')
    goal_denotation = _get_goal_denotation(goals, obj_encoding)

    # set initial state and value trace
    current_state = initial
    collated_input, encoded_states = _to_input([ current_state ], goal_denotation, obj_encoding, augment_fn, language, device, logger)
    state_trace = [ encoded_states[0] ]
    initial_values, initial_solvables = model(collated_input)
    value_trace = [ initial_values[0] + (1.0 - torch.round(torch.sigmoid(initial_solvables))[0]) * unsolvable_weight ]  # A large value indicates an unsolvable state
    if logger: logger.debug(f'initial_state={current_state}')

    # calculate greedy trace
    step, num_evaluations = 1, 1
    while (not current_state[goals]) and (len(state_trace) < max_state_trace_length):
        if cycles == 'detect' and current_state in closed_states:
            if logger:
                logger.info(colored(f"Cycle detected after last action '{action_trace[-1]}'", 'magenta'))
            break
        closed_states.add(current_state)
        if logger: logger.debug(f'**** STEP {step+1}')
        step += 1

        # explore current state (avoid loops by removing already visited successors)
        successor_candidates = [ transition for transition in _get_successor_states(current_state, actions) ]
        if cycles == 'avoid':
            successor_candidates = [ transition for transition in successor_candidates if transition[1] not in closed_states ]

        if len(successor_candidates) == 0:
            logger.info(f'No applicable action that yields unvisited state for current_state={current_state}')
            logger.info(f'Applicable actions = {_get_applicable_actions(current_state, actions)}')
            break

        successor_actions = [ candidate[0] for candidate in successor_candidates ]
        successor_states = [ candidate[1] for candidate in successor_candidates ]
        if logger: logger.debug(f'#actions={len(successor_actions)}, actions={successor_actions}')

        # calculate values for successors and best successor
        collated_input, encoded_states = _to_input(successor_states, goal_denotation, obj_encoding, augment_fn, language, device, logger)
        output_values, output_solvables = model(collated_input)
        output_values += (1.0 - torch.round(torch.sigmoid(output_solvables))) * unsolvable_weight
        best_successor_index = torch.argmin(output_values)
        num_evaluations += len(successor_actions)
        if logger:
            logger.debug(f'     values=[' + ", ".join([ f'{x[0]:.3f}' for x in output_values ]) + ']')
            logger.debug(f'  solvables=[' + ", ".join([ f'{x[0]:.3f}' for x in output_solvables ]) + ']')
            logger.debug(f'best_action={successor_actions[best_successor_index]} (index={best_successor_index})\n')

        # extend traces and set next current state
        action_trace.append(successor_actions[best_successor_index])
        state_trace.append(encoded_states[best_successor_index])
        value_trace.append(output_values[best_successor_index])
        current_state = successor_states[best_successor_index]
        if logger:
            logger.debug(f'current_state={current_state}')
            logger.debug('')

    reached_goal = current_state[goals]
    if logger: logger.debug(f'status={1 if reached_goal else 0}')
    return action_trace, state_trace, value_trace, reached_goal, num_evaluations

def compute_traces(actions, initial, goal, language, model: pl.LightningModule, cycles: str = 'avoid', max_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None):
    objects = language.constants()
    obj_encoding = create_object_encoding(objects)
    if logger: logger.info(f'{len(objects)} object(s), obj_encoding={obj_encoding}')

    with torch.no_grad():
        return policy_search(actions, initial, goal, obj_encoding, model, cycles=cycles, max_trace_length=max_state_trace_length, unsolvable_weight=unsolvable_weight, logger=logger)

def policy_search_with_augmented_states(actions, initial, goals, obj_encoding: Dict[str, int], language, model: pl.LightningModule, augment_fn = None, cycles: str = 'avoid', max_state_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None, is_spanner = False):
    device = model.device
    closed_states = set()
    action_trace = []

    # calculate denotation of goal atoms that is equal for every state
    if logger: logger.info(f'goals={goals}')
    goal_denotation = _get_goal_denotation(goals, obj_encoding)

    # set initial state and value trace
    current_state = initial
    collated_input, encoded_states = _to_input([ current_state ], goal_denotation, obj_encoding, augment_fn, language, device, logger)
    state_trace = [ encoded_states[0] ]
    initial_values, initial_solvables = model(collated_input)
    value_trace = [ initial_values[0] + (1.0 - torch.round(torch.sigmoid(initial_solvables))[0]) * unsolvable_weight ]  # A large value indicates an unsolvable state
    if logger: logger.debug(f'initial_state={current_state}')

    # calculate greedy trace
    step, num_evaluations = 1, 1
    while (not current_state[goals]) and (len(state_trace) < max_state_trace_length):
        if cycles == 'detect' and current_state in closed_states:
            if logger:
                logger.info(colored(f"Cycle detected after last action '{action_trace[-1]}'", 'magenta'))
            break
        closed_states.add(current_state)
        if logger: logger.debug(f'**** STEP {step+1}')
        step += 1

        # SPANNER: special case to avoid very time-consuming execution
        if is_spanner and _spanner_unsolvable(current_state, logger):
            logger.info(colored(f'SPANNER TASK FAILURE', 'red', attrs=[ 'bold' ]))
            break
        elif is_spanner and _spanner_solved(current_state, logger):
            logger.info(colored(f'SPANNER TASK SOLVED', 'green', attrs=[ 'bold' ]))
            logger.info(f'current_state={current_state}')
            break

        # explore current state (avoid loops by removing already visited successors)
        successor_candidates = [ transition for transition in _get_successor_states(current_state, actions) ]
        if cycles == 'avoid':
            successor_candidates = [ transition for transition in successor_candidates if transition[1] not in closed_states ]

        if len(successor_candidates) == 0:
            logger.info(f'No applicable action that yields unvisited state for current_state={current_state}')
            logger.info(f'Applicable actions = {_get_applicable_actions(current_state, actions)}')
            break

        successor_actions = [ candidate[0] for candidate in successor_candidates ]
        successor_states = [ candidate[1] for candidate in successor_candidates ]
        if logger: logger.debug(f'#actions={len(successor_actions)}, actions={successor_actions}')

        # calculate values for successors and best successor
        collated_input, encoded_states = _to_input(successor_states, goal_denotation, obj_encoding, augment_fn, language, device, logger)
        output_values, output_solvables = model(collated_input)
        output_solvables = torch.round(torch.sigmoid(output_solvables))
        #output_values += (1.0 - torch.round(torch.sigmoid(output_solvables))) * unsolvable_weight
        output_values += (1.0 - output_solvables) * unsolvable_weight
        best_successor_index = torch.argmin(output_values)
        num_evaluations += len(successor_actions)
        if logger:
            logger.debug(f'     values=[' + ", ".join([ f'{x[0]:.3f}' for x in output_values ]) + ']')
            logger.debug(f'  solvables=[' + ", ".join([ f'{x[0]:.3f}' for x in output_solvables ]) + ']')
            logger.debug(f'best_action={successor_actions[best_successor_index]} (index={best_successor_index})\n')

        # extend traces and set next current state
        value_trace.append(output_values[best_successor_index])
        state_trace.append(encoded_states[best_successor_index])
        action_trace.append(successor_actions[best_successor_index])
        current_state = successor_states[best_successor_index]

        if logger:
            logger.debug(f'current_state={current_state}')
            logger.debug('')

    reached_goal = current_state[goals]
    if logger: logger.debug(f'status={1 if reached_goal else 0}')
    return action_trace, state_trace, value_trace, reached_goal, num_evaluations

def compute_traces_with_augmented_states(actions, initial, goal, language, model: pl.LightningModule, augment_fn = None, cycles: str = 'avoid', max_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None, is_spanner = False):
    objects = language.constants()
    obj_encoding = create_object_encoding(objects)
    if logger: logger.info(f'{len(objects)} object(s), obj_encoding={obj_encoding}')

    with torch.no_grad():
        return policy_search_with_augmented_states(actions, initial, goal, obj_encoding, language, model, augment_fn=augment_fn, cycles=cycles, max_state_trace_length=max_trace_length, unsolvable_weight=unsolvable_weight, logger=logger, is_spanner=is_spanner)

