import pytorch_lightning as pl
import torch

from architecture import MaxRelationMessagePassingModel
from torch.functional import Tensor
from typing import Dict, List, Tuple


class ProblemModel(pl.LightningModule):
    def __init__(self, predicates: list, goal_predicates: list, hidden_size: int, iterations: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = MaxRelationMessagePassingModel(predicates, hidden_size, iterations)

    def forward(self, length: int, max_length: int) -> Tensor:
        node_states = self.model(states)  # TODO: Hmm! Introduce something that produces states?
        # For each predicate
        #   For each possible atom (combination of objects as arguments)
        #     Readout a Boolean if it holds, and construct an initial state s0 based on the readouts
        #
        # For each goal predicate
        #   For each possible atom (combination of objects as arguments)
        #     Readout a Boolean if it holds, and construct a goal g based on the readouts
        #
        # Find an optimal plan p=(s0, s1, ..., sN) for the problem (s0, g), N <= max_length.
        # If there is not up to size max_length, then loss is max_length.
        # Otherwise, step through the states while decrementing length. (Is this necessary?)
        # The loss is the absolute value of length (i.e. |N - length|).
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_index):
        raise NotImplementedError()

    def validation_step(self, validation_batch, batch_index):
        raise NotImplementedError()