from generators import create_object_encoding
from pathlib import Path
from torch.utils.data.dataset import Dataset
from typing import List

from generators.plan import load_pddl_problem


def _parse_problem_file(problem_file: str):
    domain_file = list(Path(problem_file).parent.glob('*domain*'))[0]
    problem = load_pddl_problem(domain_file, problem_file)
    object_encoding = create_object_encoding(problem['objects'])
    return (problem, object_encoding)


class OnlineDataset(Dataset):
    def __init__(self, problem_files: List[str]):
        self._problem_files = problem_files

    def __len__(self):
        return len(self._problem_files)

    def __getitem__(self, idx):
        problem_file = self._problem_files[idx]
        return _parse_problem_file(problem_file)


def load_dataset(problem_directory: Path, not_used_1: int, not_used_2: int):
    problem_files = [str(pddl_file) for pddl_file in problem_directory.glob('*.pddl') if 'domain' not in pddl_file.name]
    # Pick one problem and read the predicates from the domain. This could be optimized but does not matter.
    problem, _ = _parse_problem_file(problem_files[0])
    predicate_arities = [(predicate.name, predicate.arity) for predicate in problem['predicates']]
    predicate_arities_with_goals = predicate_arities + [(predicate + '_goal', arity) for predicate, arity in predicate_arities]
    return (OnlineDataset(problem_files), predicate_arities_with_goals)


def collate(batch):
    """
    Input: [(task, encoding)]
    Output: [(task, encoding)]
    """
    return batch
