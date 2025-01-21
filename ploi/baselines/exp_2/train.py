import argparse
import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
import torch
import pymimir as mm
import multiprocessing as mp
from typing import Dict, List, Tuple,Union, Optional
from collections import defaultdict
import random
import os
import time
from datetime import timedelta
from torch.nn.functional import l1_loss
#from architecture.supervised.optimal import MaxModel, AddModel
from ploi.baselines.exp_2.architecture.supervised.optimal import MaxModel, AddModel
from ploi.baselines.exp_2.datasets.supervised.optimal.dataset import create_dataset_from_state_spaces
from ploi.baselines.exp_3.helpers import ValidationLossLogging

#from ploi.baselines.exp_2.test import test_model

def _generate_state_spaces(domain_path: str, problem_paths: List[str]) -> List[mm.StateSpace]:
    print('Generating state spaces...')
    state_spaces: List[mm.StateSpace] = []
    #for problem_path in problem_paths:
    for problem_path in problem_paths:
        print(f'> Expanding: {problem_path}')
        state_space = mm.StateSpace.create(domain_path, problem_path, mm.StateSpaceOptions(max_num_states=1_000_000, timeout_ms=60_000))
        #state_space = mm.StateSpace.create(domain_path, problem_path, mm.StateSpaceOptions(max_num_states=10_000_000, timeout_ms=300_000))
        if state_space is not None:
            state_spaces.append(state_space)
            print(f'- # States: {state_space.get_num_vertices()}')
        else:
            print('- Skipped')
    state_spaces.sort(key=lambda state_space: state_space.get_num_vertices())
    return state_spaces

class StateSampler:
    def __init__(self, state_spaces: List[mm.StateSpace], max_samples_per_cost: int = 100) -> None:
        self._state_spaces = state_spaces
        self._max_distances = []
        self._has_deadends = []
        self._deadend_distance = float('inf')
        self._max_samples_per_cost = max_samples_per_cost

        # Initialize tracking structures
        self._samples_per_cost = defaultdict(int)  # Track number of samples per cost
        self._sampled_states = defaultdict(set)    # Track unique states per cost

        # Track total available states
        self._total_states = self._count_total_states()
        self._total_sampled_states = 0

        for state_space in state_spaces:
            max_goal_distance = 0
            has_deadend = False
            for goal_distance in state_space.get_goal_distances():
                if goal_distance != self._deadend_distance:
                    max_goal_distance = max(max_goal_distance, int(goal_distance))
                else:
                    has_deadend = True
            self._max_distances.append(max_goal_distance)
            self._has_deadends.append(has_deadend)

    def sample_old(self) -> Tuple[mm.State, mm.StateSpace, int]:
        # To achieve an even distribution, we uniformly sample a state space and select a valid goal-distance within that space.
        # Finally, we randomly sample a state from the selected state space and with the goal-distance.
        state_space_index = random.randint(0, len(self._state_spaces) - 1)
        sampled_state_space = self._state_spaces[state_space_index]
        max_goal_distance = self._max_distances[state_space_index]
        has_deadends = self._has_deadends[state_space_index]
        goal_distance = random.randint(-1 if has_deadends else 0, max_goal_distance)
        if goal_distance < 0:
            sampled_state_index = sampled_state_space.sample_vertex_index_with_goal_distance(self._deadend_distance)
        else:
            sampled_state_index = sampled_state_space.sample_vertex_index_with_goal_distance(goal_distance)
        sampled_state = sampled_state_space.get_vertex(sampled_state_index)
        return (sampled_state.get_state(), sampled_state_space, goal_distance)

    def sample(self) -> Tuple[mm.State, mm.StateSpace, int]:
        """
        Sample a state while respecting the maximum samples per cost constraint.
        
        Returns:
            Tuple of (State, StateSpace, goal_distance)
            
        Raises:
            RuntimeError: If unable to find a valid sample after max attempts
        """
        max_attempts = 1000  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # Sample state space and cost
            state_space_index = random.randint(0, len(self._state_spaces) - 1)
            sampled_state_space = self._state_spaces[state_space_index]
            max_goal_distance = self._max_distances[state_space_index]
            has_deadends = self._has_deadends[state_space_index]
            
            # Sample goal distance
            goal_distance = random.randint(-1 if has_deadends else 0, max_goal_distance)
            
            # Check if we've reached the maximum samples for this cost
            if self._samples_per_cost[goal_distance] >= self._max_samples_per_cost:
                attempts += 1
                continue
                
            # Sample state
            if goal_distance < 0:
                sampled_state_index = sampled_state_space.sample_vertex_index_with_goal_distance(
                    self._deadend_distance
                )
            else:
                sampled_state_index = sampled_state_space.sample_vertex_index_with_goal_distance(
                    goal_distance
                )
                
            sampled_state = sampled_state_space.get_vertex(sampled_state_index)
            state = sampled_state.get_state()
            
            # Check if this exact state was already sampled for this cost
            state_hash = hash(state)  # Create a hashable representation of the state
            if state_hash in self._sampled_states[goal_distance]:
                attempts += 1
                continue
                
            # Update tracking structures
            self._samples_per_cost[goal_distance] += 1
            self._sampled_states[goal_distance].add(state_hash)
            
            return (state, sampled_state_space, goal_distance)
            
        raise RuntimeError(
            f"Unable to find valid sample after {max_attempts} attempts. "
            "Consider increasing max_samples_per_cost or checking state space coverage."
        )

    def _count_total_states(self) -> int:
        """Count total unique states across all state spaces."""
        return sum(space.get_num_vertices() for space in self._state_spaces)

    def is_sampling_complete(self) -> bool:
        """
        Check if sampling is complete based on either condition:
        1. All unique states have been sampled
        2. All costs have reached their maximum samples
        """
        if self._total_sampled_states >= self._total_states:
            return True

        # First check if we've sampled anything at all
        if not self._samples_per_cost:
            return False
            
        return all(count >= self._max_samples_per_cost 
                  for count in self._samples_per_cost.values())

    def sample_batch(self, batch_size: int, device: torch.device) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        if self.is_sampling_complete():
            return None
            
        relations = {}
        sizes = []
        targets = []
        
        successful_samples = 0
        max_attempts = batch_size * 10
        attempts = 0
        
        while successful_samples < batch_size and attempts < max_attempts:
            try:
                state, state_space, target = self.sample()
                self._sample_state_to_batch(state, state_space, target, relations, sizes, targets)
                successful_samples += 1
            except RuntimeError:
                attempts += 1
                if attempts >= max_attempts:
                    if successful_samples == 0:
                        return None
                    break
        
        relation_tensors = relations_to_tensors(relations, device)
        size_tensor = torch.tensor(sizes, dtype=torch.int, device=device, requires_grad=False)
        target_tensor = torch.tensor(targets, dtype=torch.float, device=device, requires_grad=False)
        
        return relation_tensors, size_tensor, target_tensor

    def _sample_state_to_batch(
        self,
        state: mm.State,
        state_space: mm.StateSpace,
        target: int,
        relations: Dict[str, List[int]],
        sizes: List[int],
        targets: List[int]
    ) -> None:
        offset = sum(sizes)

        def add_relations(atom, is_goal_atom):
            predicate_name = get_atom_name(atom, state, is_goal_atom)
            term_ids = [term.get_index() + offset for term in atom.get_objects()]
            if predicate_name not in relations:
                relations[predicate_name] = term_ids
            else:
                relations[predicate_name].extend(term_ids)

        for atom in get_atoms(state, state_space.get_problem(), state_space.get_pddl_repositories()):
            add_relations(atom, False)
        
        for atom in get_goal(state_space.get_problem()):
            add_relations(atom, True)

        sizes.append(len(state_space.get_problem().get_objects()))
        targets.append(target)

def _create_state_samplers(state_spaces: List[mm.StateSpace]) -> Tuple[StateSampler, StateSampler]:
    print('Creating state samplers...')
    train_size = int(len(state_spaces) * 0.8)
    train_state_spaces = state_spaces[:train_size]
    validation_state_spaces = state_spaces[train_size:]
    train_dataset = StateSampler(train_state_spaces)
    validation_dataset = StateSampler(validation_state_spaces)
    return train_dataset, validation_dataset

def create_state_samplers_separate(train_state_spaces: List[mm.StateSpace],
                                   val_state_spaces: List[mm.StateSpace],
                                   ) -> Tuple[StateSampler, StateSampler]:
    train_dataset = StateSampler(train_state_spaces)
    validation_dataset = StateSampler(val_state_spaces)
    return train_dataset, validation_dataset

def _parse_instances(input: Path) -> Tuple[str, List[str]]:
    print('Parsing files...')
    if input.is_file():
        #domain_file = str(input.parent / 'domain.pddl')
        #CHECK - NOT TESTED 
        domain_filename = str(input).split("/")[-2][:-1] + ".pddl"
        domain_file = Path("/".join(str(input).split("/")[:-2]) + "/" +domain_filename)
        problem_files = [str(input)]
    else:
        #domain_file = str(input / 'domain.pddl')
        domain_filename = str(input).split("/")[-1] + ".pddl"
        domain_file = "/".join(str(input).split("/")[:-1]) + "/" + domain_filename
        if Path(domain_file).exists() is False :
            domain_file = str(input / 'domain.pddl')
        problem_files = [str(file) for file in input.glob('*.pddl') if file.name != 'domain.pddl']
        problem_files.sort()
    return domain_file, problem_files

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, type=Path, help='Path to training dataset')
    parser.add_argument('--validation', required=True, type=Path, help='Path to validation dataset')
    parser.add_argument('--size', default=32, type=int, help='Number of features per object')
    parser.add_argument('--iterations', default=30, type=int, help='Number of convolutions')
    parser.add_argument('--batch_size', default=16, type=int, help='Maximum size of batches')
    parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='Learning rate of training session')
    parser.add_argument('--l1', default=0.0, type=float, help='Strength of L1 regularization')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Strength of weight decay regularization')
    parser.add_argument('--gradient_accumulation', default=1, type=int, help='Number of gradients to accumulate before step')
    parser.add_argument('--max_samples_per_value', default=100, type=int, help='Maximum number of states per value per dataset')
    parser.add_argument('--patience', default=40, type=int, help='Patience for early stopping')
    parser.add_argument('--gradient_clip', default=0.1, type=float, help='Gradient clip value')
    parser.add_argument('--profiler', default=None, type=str, help='"simple", "advanced" or "pytorch"')
    parser.add_argument('--verbose', action='store_true', help='Print additional information during training')
    parser.add_argument('--separate_train_val', action='store_true', help='If training and validation datasets are from different folders')
    parser.add_argument('--train_only', action='store_true', help='Train only')
    parser.add_argument('--test_only', action='store_true', help='Test only')
    parser.add_argument('--train_test', action='store_true', help='run both testing and training')
    parser.add_argument('--run_original', action='store_true', help='run original setting')
    parser.add_argument('--max_epochs', default=1000, help='Max epochs for training')
    parser.add_argument('--domain', required=True,  help='domain being trained')
    parser.add_argument('--model_type', default="max", help='model type being trained')

    args = parser.parse_args()
    return args

def get_predicate_name(predicate: Union[mm.StaticPredicate, mm.FluentPredicate, mm.DerivedPredicate], is_goal_predicate: bool, is_true: bool):
    assert (not is_goal_predicate and is_true) or (is_goal_predicate)
    if is_goal_predicate: return ('relation_' + predicate.get_name() + '_goal') + ('_true' if is_true else '_false')
    else: return 'relation_' + predicate.get_name()

def get_predicates(domain):
    predicates = []
    predicates.extend(domain.get_static_predicates())
    predicates.extend(domain.get_fluent_predicates())
    predicates.extend(domain.get_derived_predicates())
    relation_name_arities = [(get_predicate_name(predicate, False, True), len(predicate.get_parameters())) for predicate in predicates]
    relation_name_arities.extend([(get_predicate_name(predicate, True, True), len(predicate.get_parameters())) for predicate in predicates])
    relation_name_arities.extend([(get_predicate_name(predicate, True, False), len(predicate.get_parameters())) for predicate in predicates])

    return relation_name_arities 

def _load_datasets(args):
    print('Loading datasets...')
    from ploi.baselines.exp_2.datasets.supervised.optimal import load_dataset, collate
    (train_dataset, predicates) = load_dataset(args.train, args.max_samples_per_value)
    (validation_dataset, _) = load_dataset(args.validation, args.max_samples_per_value)
    num_workers = mp.cpu_count() - 2
    loader_params = {
        "batch_size": args.batch_size,
        "drop_last": False,
        "collate_fn": collate,
        "pin_memory": True,
        "num_workers": num_workers
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_params)
    return predicates, train_loader, validation_loader

def _load_datasets_v2(args):
    from ploi.baselines.exp_2.datasets.supervised.optimal import collate
    domain_path, problem_paths = _parse_instances(args.train)
    state_spaces = _generate_state_spaces(domain_path, problem_paths)

    if args.separate_train_val is True :
        domain_path_val, problem_paths_val = _parse_instances(args.validation)
        val_state_spaces = _generate_state_spaces(domain_path_val, problem_paths_val)
        #train_dataset = create_dataset_from_state_spaces(state_spaces)
        #validation_dataset = create_dataset_from_state_spaces(val_state_spaces) 
        train_dataset, validation_dataset = create_state_samplers_separate(state_spaces, val_state_spaces)
    else :
        train_dataset, validation_dataset = _create_state_samplers(state_spaces)
        #train_size = int(len(state_spaces) * 0.8)
        #train_state_spaces = state_spaces[:train_size]
        #val_state_spaces = state_spaces[train_size:]

        #train_dataset = create_dataset_from_state_spaces(train_state_spaces, args.max_samples_per_value)
        #validation_dataset = create_dataset_from_state_spaces(val_state_spaces, args.max_samples_per_value) 

    domain = state_spaces[0].get_problem().get_domain()
    predicates = get_predicates(domain)

    '''
    num_workers = mp.cpu_count() - 2

    loader_params = {
        "batch_size": args.batch_size,
        "drop_last": False,
        "collate_fn": collate,
        "pin_memory": True,
        "num_workers": num_workers
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_params)
    '''

    #return predicates, train_loader, validation_loader
    #train_loader, validation_loader = create_loaders_from_datasets(train_dataset, validation_dataset)
    return predicates, train_dataset, validation_dataset

def _initialize_model(args, predicates):
    print('Loading model...')
    model_params = {
        "predicates": predicates,
        "hidden_size": args.size,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "l1_factor": args.l1,
        "weight_decay": args.weight_decay,
    }
    from ploi.baselines.exp_2.architecture.supervised.optimal import MaxModel as Model
    model = Model(**model_params)
    return model

def _load_trainer(args):
    print('Initializing trainer...')
    callbacks = []
    if not args.verbose: callbacks.append(ValidationLossLogging())
    early_stopping = EarlyStopping(monitor='validation_loss', patience=args.patience)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor='validation_loss')
    callbacks.append(early_stopping)
    callbacks.append(checkpoint)
    trainer_params = {
        "num_sanity_val_steps": 0,
        #"progress_bar_refresh_rate": 30 if args.verbose else 0,
        "callbacks": callbacks,
        #"weights_summary": None,
        #"auto_lr_find": True,
        "profiler": args.profiler,
        "accumulate_grad_batches": args.gradient_accumulation,
        "gradient_clip_val": args.gradient_clip,
    }
    #if args.gpus > 0: trainer = pl.Trainer(gpus=args.gpus, auto_select_gpus=True, **trainer_params)
    #else: trainer = pl.Trainer(**trainer_params)
    if args.gpus > 0:
        trainer = pl.Trainer(accelerator="gpu",devices=args.gpus,
            **trainer_params)
    else:
        trainer = pl.Trainer(accelerator="cpu", **trainer_params)
    return trainer

def get_atoms(state: mm.State, problem: mm.Problem, factories: mm.PDDLRepositories) -> List[Union[mm.StaticGroundAtom, mm.FluentGroundAtom, mm.DerivedGroundAtom]]:
    atoms = [literal.get_atom() for literal in problem.get_static_initial_literals()]
    atoms.extend(factories.get_fluent_ground_atoms_from_indices(state.get_fluent_atoms()))
    atoms.extend(factories.get_derived_ground_atoms_from_indices(state.get_derived_atoms()))
    return atoms

def get_atom_name(atom: Union[mm.StaticAtom, mm.FluentAtom, mm.DerivedAtom], state: mm.State, is_goal_atom: bool):
    if is_goal_atom: return get_predicate_name(atom.get_predicate(), True, state.contains(atom))
    else: return get_predicate_name(atom.get_predicate(), False, True)

def get_goal(problem: mm.Problem) -> List[Union[mm.StaticGroundAtom, mm.FluentGroundAtom, mm.DerivedGroundAtom]]:
    static_goal = [literal.get_atom() for literal in problem.get_static_goal_condition()]
    fluent_goal = [literal.get_atom() for literal in problem.get_fluent_goal_condition()]
    derived_goal = [literal.get_atom() for literal in problem.get_derived_goal_condition()]
    full_goal = static_goal + fluent_goal + derived_goal
    return full_goal

def relations_to_tensors(term_id_groups: Dict[str, List[int]], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in term_id_groups.items():
        result[key] = torch.tensor(value, dtype=torch.int, device=device, requires_grad=False)
    return result

def _sample_state_to_batch(relations: Dict[str, List[int]], sizes: List[int], targets: List[int], states: StateSampler):
    state, state_space, target = states.sample()
    offset = sum(sizes)
    # Helper function for populating relations and sizes.
    def add_relations(atom, is_goal_atom):
        predicate_name = get_atom_name(atom, state, is_goal_atom)
        term_ids = [term.get_index() + offset for term in atom.get_objects()]
        if predicate_name not in relations: relations[predicate_name] = term_ids
        else: relations[predicate_name].extend(term_ids)
    # Add state to relations and sizes, together with the goal.
    for atom in get_atoms(state, state_space.get_problem(), state_space.get_pddl_repositories()): add_relations(atom, False)
    for atom in get_goal(state_space.get_problem()): add_relations(atom, True)
    sizes.append(len(state_space.get_problem().get_objects()))
    targets.append(target)

def _sample_batch(states: StateSampler, batch_size: int, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    relations = {}
    sizes = []
    targets = []
    for _ in range(batch_size):
        _sample_state_to_batch(relations, sizes, targets, states)
    relation_tensors = relations_to_tensors(relations, device)
    size_tensor = torch.tensor(sizes, dtype=torch.int, device=device, requires_grad=False)
    target_tensor = torch.tensor(targets, dtype=torch.float, device=device, requires_grad=False)
    return relation_tensors, size_tensor, target_tensor

def train_model(model, train_states, validation_states, args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus > 0 else "cpu")
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=model.learning_rate, 
                               weight_decay=model.weight_decay)
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print('Creating datasets...')
    #train_dataset = [_sample_batch(train_states, args.batch_size, device) for _ in range(10_000)]
    #train_dataset = [_sample_batch(train_states, args.batch_size, device) for _ in range(1_000)]
    #validation_dataset = [_sample_batch(validation_states, args.batch_size, device) for _ in range(1_000)]
    # Create the dataset
    train_dataset = create_batch_dataset(
        sampler=train_states,
        batch_size=args.batch_size,
        device=device
    )

    validation_dataset = create_batch_dataset(
        sampler=validation_states,
        batch_size=args.batch_size,
        device=device
    )

    start_time = time.time()
    epoch_start_time = None
    
    for epoch in range(int(args.max_epochs)):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        for batch_idx, (states, sizes, target) in enumerate(train_dataset):
            #states, sizes,  target = states.to(device), sizes.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model((states,sizes))
            loss = torch.mean(torch.abs(torch.sub(target, output)))
            
            # L1 regularization
            if model.l1_factor > 0.0:
                l1_loss_val = 0.0
                for parameter in model.parameters():
                    l1_loss_val += torch.sum(model.l1_factor * torch.abs(parameter))
                loss += l1_loss_val
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if args.verbose and batch_idx % 1000 == 0:
                elapsed = time.time() - start_time
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f} | '
                      f'Time elapsed: {timedelta(seconds=int(elapsed))}')
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for states, sizes, target in validation_dataset:
                output = model((states,sizes))
                val_loss = l1_loss(output, target)
                val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            save_model(model, optimizer, epoch, args)
        else:
            patience_counter += 1
        
        # Print epoch stats
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        print(f'Epoch {epoch + 1}/{args.max_epochs} completed in {timedelta(seconds=int(epoch_time))} | '
              f'Validation Loss: {avg_val_loss:.4f} | '
              f'Best Val Loss: {best_val_loss:.4f} | '
              f'Total Training Time: {timedelta(seconds=int(total_time))}')
            
        if patience_counter >= args.patience:
            print(f'Early stopping triggered after {epoch + 1} epochs | '
                  f'Total Training Time: {timedelta(seconds=int(time.time() - start_time))}')
            break


    total_training_time = time.time() - start_time
    print(f'\nTraining completed in {timedelta(seconds=int(total_training_time))}')
    print(f'Best validation loss: {best_val_loss:.4f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, optimizer

def create_batch_dataset(sampler: StateSampler, batch_size: int, device: str) -> List:
    """
    Create training dataset by sampling until completion.
    
    Args:
        sampler: Initialized BalancedStateSampler
        batch_size: Size of each batch
        device: Device to put the tensors on
        
    Returns:
        List of all valid batches
    """
    batch_dataset = []
    
    while True:
        batch = sampler.sample_batch(batch_size, device)
        if batch is None:
            break
        batch_dataset.append(batch)
        
        # Optional: Print progress
        if len(batch_dataset) % 100 == 0:
            print(f"Collected {len(batch_dataset)} batches")
            print(f"Samples per cost: {sampler.get_samples_distribution()}")
    
    return batch_dataset

def train(args):

    predicates, train_loader, validation_loader = _load_datasets_v2(args)
    model = _initialize_model(args, predicates)

    #loaded_model = load_model(model, "models/Manyblocks_ipcc_big_exp_2/model_best.pth", MaxModel)

    # Train the model
    trained_model, optimizer = train_model(model, train_loader, validation_loader, args)

    # Save the trained model

    #trained_model = None
    #optimizer = None
    save_model(trained_model, optimizer, "best", args)
    return trained_model

def train_original(args):
    predicates, train_loader, validation_loader = _load_datasets(args)
    model = _initialize_model(args, predicates)

    # Train the model
    model = _load_model_original(args, predicates)
    trainer = _load_trainer(args)
    print('Training model...')
    trainer.fit(model, train_loader, validation_loader)

def _load_model_original(args, predicates):
    print('Loading model...')
    model_params = {
        "predicates": predicates,
        "hidden_size": args.size,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "l1_factor": args.l1,
        "weight_decay": args.weight_decay,
    }
    from ploi.baselines.exp_2.architecture.supervised.optimal import MaxModel as Model
    model = Model(**model_params)
    return model


def save_model(model, optimizer, epoch, args ):
    """Save model checkpoint with additional training info."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hparams' : model.get_hparams_dict(),
        'epoch': epoch,
    }

    # Get current directory
    current_dir = os.getcwd()

    # Create models directory path
    models_dir = os.path.join(current_dir, 'models')
    curr_domain_dir = os.path.join(models_dir, args.domain + "_exp_2")

    # Create the models directory if it doesn't exist
    os.makedirs(curr_domain_dir, exist_ok=True)
    model_path = os.path.join(curr_domain_dir, 'model_' + str(epoch) + '.pth')

    torch.save(checkpoint, model_path)

def load_model(path, device, model_class):
    """Load model checkpoint with additional training info."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hparams_dict = checkpoint['hparams']
    model = model_class(**hparams_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=model.learning_rate, 
                               weight_decay=model.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer 

def _plan_exp_2(problem: mm.Problem, factories: mm.PDDLRepositories, 
          model: MaxModel, device: torch.device,
          max_plan_length = 1000) -> Union[None, List[mm.GroundAction]]:
    solution = []
    visited_states = []
    # Helper function for testing is a state is a goal state.
    def is_goal_state(state: mm.State) -> bool:
        return state.literals_hold(problem.get_fluent_goal_condition()) and state.literals_hold(problem.get_derived_goal_condition())
    # Disable gradient as we are not optimizing.
    with torch.no_grad():
        successor_generator = mm.LiftedApplicableActionGenerator(problem, factories)
        state_repository = mm.StateRepository(successor_generator)
        current_state = state_repository.get_or_create_initial_state()
        visited_states.append(current_state)
        while (not is_goal_state(current_state)) and (len(solution) < max_plan_length):
            applicable_actions = list(successor_generator.compute_applicable_actions(current_state))
            successor_states = [state_repository.get_or_create_successor_state(current_state, action)[0] for action in applicable_actions]
            relations, sizes = create_input(problem, successor_states, factories, device)
            values = model.forward((relations, sizes))
            # TODO: Take deadends into account.

            min_successor = None 
            min_index = -1
            while len(successor_states) > 0 :
                min_index = values.argmin()
                min_action = applicable_actions[min_index]
                min_successor = successor_states[min_index]
                current_state = min_successor

                if min_successor in visited_states:
                    values = torch.cat([values[:min_index], values[min_index+1:]])
                    applicable_actions = applicable_actions[:min_index] + applicable_actions[min_index+1:]
                    successor_states = successor_states[:min_index] + successor_states[min_index+1:]
                else :
                    break

            if len(successor_states) == 0:
                return None

            solution.append(min_action)
            visited_states.append(current_state)
            #print(f'{min_value.item():.3f}: {min_action.to_string_for_plan(factories)}')
    return solution if is_goal_state(current_state) else None 

def test_model(args):
    #path = "models/" + args.domain + "_exp_2/model_best.pth"

    if args.model_type == "max":
        model_class = MaxModel
    else :
        model_class = AddModel

    if args.test_train is True :
        test_path = "models/" + args.domain + "_exp_2/model_best.pth"
    else :
        test_path = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus > 0 else "cpu")
    model = load_model(test_path, device, model_class)


def create_input(problem: mm.Problem, states: List[mm.State], factories: mm.PDDLRepositories, device: torch.device):
    relations = {}
    sizes = []
    # Helper function for populating relations and sizes.
    def add_relations(atom, offset, is_goal_atom):
        predicate_name = get_atom_name(atom, state, is_goal_atom)
        term_ids = [term.get_index() + offset for term in atom.get_objects()]
        if predicate_name not in relations: relations[predicate_name] = term_ids
        else: relations[predicate_name].extend(term_ids)
    # Add all states to relations and sizes, together with the goal.
    for state in states:
        offset = sum(sizes)
        for atom in get_atoms(state, problem, factories): add_relations(atom, offset, False)
        for atom in get_goal(problem): add_relations(atom, offset, True)
        sizes.append(len(problem.get_objects()))
    # Move all lists to the GPU as tensors.
    return relations_to_tensors(relations, device), torch.tensor(sizes, dtype=torch.int, device=device, requires_grad=False)

if __name__ == "__main__":
    args = _parse_arguments()
    args.domain = args.domain.capitalize()

    if args.run_original is True:
        train_original(args)
        exit()

    if args.train_only is True:
        train(args)
        exit()

    if args.test_only is True:
        test_model(args)
        exit()

    if args.train_test is True :
        model = train(args)
        test_model(args)
        exit()