import torch
import torch.nn as nn
import pytorch_lightning as pl

from architecture import AddModelBase, MaxModelBase, MaxReadoutModelBase, AddMaxModelBase
from architecture import supervised_optimal_loss, selfsupervised_optimal_loss, selfsupervised_suboptimal_loss, selfsupervised_suboptimal2_loss, unsupervised_optimal_loss, unsupervised_suboptimal_loss, l1_regularization
from architecture.attention_base import AttentionModelBase
from generators.plan import policy_search

_max_trace_length = 4

def set_max_trace_length(max_length: int):
    global _max_trace_length
    _max_trace_length = max_length

def _create_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def _create_supervised_model_class(base: pl.LightningModule, loss):
    """Create a model class for supervised training that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay

        def configure_optimizers(self):
            return _create_optimizer(self, self.learning_rate, self.weight_decay)

        def training_step(self, train_batch, batch_index):
            states, target = train_batch
            output = self(states)
            train = loss(output, target)
            self.log('train_loss', train)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train + l1
            self.log('total_loss', total)
            return total

        def validation_step(self, validation_batch, batch_index):
            states, target = validation_batch
            output = self(states)
            validation = loss(output, target)
            self.log('validation_loss', validation)

    return Model

def _create_unsupervised_model_class(base: pl.LightningModule, loss):
    """Create a model class for unsupervised training that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay

        def configure_optimizers(self):
            return _create_optimizer(self, self.learning_rate, self.weight_decay)

        def training_step(self, train_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = train_batch
            output = self(collated_states_with_object_counts)
            train = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('train_loss', train)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train + l1
            self.log('total_loss', total)
            return total

        def validation_step(self, validation_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = validation_batch
            output = self(collated_states_with_object_counts)
            validation = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('validation_loss', validation)

    return Model

def _create_online_model_class(base: pl.LightningModule, loss):
    """Create a model class for online training (unsupervised) that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay

        def configure_optimizers(self):
            return _create_optimizer(self, self.learning_rate, self.weight_decay)

        def training_step(self, train_batch, batch_index):
            train = 0.0
            for problem, encoding in train_batch:
                action_trace, state_trace, value_trace, is_solution = policy_search(problem['actions'], problem['initial'], problem['goal'], encoding, self, _max_trace_length)
                values = []
                goals = []
                counts = []
                for index in range(len(value_trace) - 1):
                    values.append(value_trace[index])
                    values.append(value_trace[index + 1])
                    goals.append(False)
                    counts.append(2)
                if is_solution:
                    values.append(value_trace[-1])
                    goals.append(True)
                    counts.append(1)
                train += loss(torch.stack(values), goals, counts, self.device)
            train /= len(train_batch)
            self.log('train_loss', train)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train + l1
            self.log('total_loss', total)
            return total

        def validation_step(self, validation_batch, batch_index):
            validation = 0.0
            for problem, encoding in validation_batch:
                action_trace, state_trace, value_trace, is_solution = policy_search(problem['actions'], problem['initial'], problem['goal'], encoding, self, _max_trace_length)
                values = []
                goals = []
                counts = []
                for index in range(len(value_trace) - 1):
                    values.append(value_trace[index])
                    values.append(value_trace[index + 1])
                    goals.append(False)
                    counts.append(2)
                if is_solution:
                    values.append(value_trace[-1])
                    goals.append(True)
                    counts.append(1)
                validation += loss(torch.stack(values), goals, counts, self.device)
            validation /= len(validation_batch)
            self.log('validation_loss', validation)

    return Model

SupervisedOptimalAddModel = _create_supervised_model_class(AddModelBase, supervised_optimal_loss)
SupervisedOptimalMaxModel = _create_supervised_model_class(MaxModelBase, supervised_optimal_loss)
SupervisedOptimalAddMaxModel = _create_supervised_model_class(AddMaxModelBase, supervised_optimal_loss)
SupervisedOptimalMaxReadoutModel = _create_supervised_model_class(MaxReadoutModelBase, supervised_optimal_loss)
SupervisedOptimalAttentionModel = _create_supervised_model_class(AttentionModelBase, supervised_optimal_loss)

SelfsupervisedOptimalAddModel = _create_unsupervised_model_class(AddModelBase, selfsupervised_optimal_loss)
SelfsupervisedOptimalMaxModel = _create_unsupervised_model_class(MaxModelBase, selfsupervised_optimal_loss)
SelfsupervisedOptimalAddMaxModel = _create_unsupervised_model_class(AddMaxModelBase, selfsupervised_optimal_loss)
SelfsupervisedOptimalMaxReadoutModel = _create_unsupervised_model_class(MaxReadoutModelBase, selfsupervised_optimal_loss)
SelfsupervisedOptimalAttentionModel = _create_unsupervised_model_class(AttentionModelBase, selfsupervised_optimal_loss)

SelfsupervisedSuboptimalAddModel = _create_unsupervised_model_class(AddModelBase, selfsupervised_suboptimal_loss)
SelfsupervisedSuboptimalMaxModel = _create_unsupervised_model_class(MaxModelBase, selfsupervised_suboptimal_loss)
SelfsupervisedSuboptimalAddMaxModel = _create_unsupervised_model_class(AddMaxModelBase, selfsupervised_suboptimal_loss)
SelfsupervisedSuboptimalMaxReadoutModel = _create_unsupervised_model_class(MaxReadoutModelBase, selfsupervised_suboptimal_loss)
SelfsupervisedSuboptimalAttentionModel = _create_unsupervised_model_class(AttentionModelBase, selfsupervised_suboptimal_loss)

SelfsupervisedSuboptimalAddModel2 = _create_unsupervised_model_class(AddModelBase, selfsupervised_suboptimal2_loss)
SelfsupervisedSuboptimalMaxModel2 = _create_unsupervised_model_class(MaxModelBase, selfsupervised_suboptimal2_loss)
SelfsupervisedSuboptimalAddMaxModel2 = _create_unsupervised_model_class(AddMaxModelBase, selfsupervised_suboptimal2_loss)
SelfsupervisedSuboptimalMaxReadoutModel2 = _create_unsupervised_model_class(MaxReadoutModelBase, selfsupervised_suboptimal2_loss)

UnsupervisedOptimalAddModel = _create_unsupervised_model_class(AddModelBase, unsupervised_optimal_loss)
UnsupervisedOptimalMaxModel = _create_unsupervised_model_class(MaxModelBase, unsupervised_optimal_loss)
UnsupervisedOptimalAddMaxModel = _create_unsupervised_model_class(AddMaxModelBase, unsupervised_optimal_loss)
UnsupervisedOptimalMaxReadoutModel = _create_unsupervised_model_class(MaxReadoutModelBase, unsupervised_optimal_loss)
UnsupervisedOptimalAttentionModel = _create_unsupervised_model_class(AttentionModelBase, unsupervised_optimal_loss)

UnsupervisedSuboptimalAddModel = _create_unsupervised_model_class(AddModelBase, unsupervised_suboptimal_loss)
UnsupervisedSuboptimalMaxModel = _create_unsupervised_model_class(MaxModelBase, unsupervised_suboptimal_loss)
UnsupervisedSuboptimalAddMaxModel = _create_unsupervised_model_class(AddMaxModelBase, unsupervised_suboptimal_loss)
UnsupervisedSuboptimalMaxReadoutModel = _create_unsupervised_model_class(MaxReadoutModelBase, unsupervised_suboptimal_loss)
UnsupervisedSuboptimalAttentionModel = _create_unsupervised_model_class(AttentionModelBase, unsupervised_suboptimal_loss)

OnlineOptimalAddModel = _create_online_model_class(AddModelBase, unsupervised_optimal_loss)
OnlineOptimalMaxModel = _create_online_model_class(MaxModelBase, unsupervised_optimal_loss)
OnlineOptimalAddMaxModel = _create_online_model_class(AddMaxModelBase, unsupervised_optimal_loss)
OnlineOptimalMaxReadoutModel = _create_online_model_class(MaxReadoutModelBase, unsupervised_optimal_loss)
OnlineOptimalAttentionModel = _create_online_model_class(AttentionModelBase, unsupervised_optimal_loss)
