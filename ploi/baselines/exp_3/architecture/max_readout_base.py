import torch
import torch.nn as nn
import pytorch_lightning as pl

# Imports related to type annotations
from typing import List, Dict, Tuple
from torch.nn.functional import Tensor


class RelationMessagePassing(nn.Module):
    def __init__(self, relations: List[Tuple[int, int]], hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.relation_modules = nn.ModuleList()
        for relation, arity in relations:
            assert relation == len(self.relation_modules)
            input_size = arity * hidden_size
            output_size = arity * hidden_size
            if (input_size > 0) and (output_size > 0):
                mlp = nn.Sequential(nn.Linear(input_size, input_size, True), nn.ReLU(), nn.Linear(input_size, output_size, True))
            else:
                mlp = None
            self.relation_modules.append(mlp)
        self.update = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size, True), nn.ReLU(), nn.Linear(2 * hidden_size, hidden_size, True))
        self.dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy.device

    def forward(self, node_states: Tensor, relations: Dict[int, Tensor]) -> Tensor:
        # Compute an aggregated message for each recipient
        max_outputs = []
        outputs = []
        for relation, module in enumerate(self.relation_modules):
            if (module is not None) and (relation in relations):
                values = relations[relation]
                input = torch.index_select(node_states, 0, values).view(-1, module[0].in_features)
                output = module(input).view(-1, self.hidden_size)
                max_outputs.append(torch.max(output))
                node_indices = values.view(-1, 1).expand(-1, self.hidden_size)
                outputs.append((output, node_indices))

        max_offset = torch.max(torch.stack(max_outputs))
        exps_sum = torch.full_like(node_states, 1E-16, device=self.get_device())
        for output, node_indices in outputs:
            exps = torch.exp(8.0 * (output - max_offset))
            exps_sum = torch.scatter_add(exps_sum, 0, node_indices, exps)

        # Update states with aggregated messages
        max_msg = ((1.0 / 8.0) * torch.log(exps_sum)) + max_offset
        next_node_states = self.update(torch.cat([max_msg, node_states], dim=1))
        return next_node_states


class Readout(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(input_size, input_size, bias), nn.ReLU(), nn.Linear(input_size, input_size, bias))
        self.post = nn.Sequential(nn.Linear(input_size, input_size, bias), nn.ReLU(), nn.Linear(input_size, output_size, bias))
        self.dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy.device

    def forward(self, batch_num_objects: List[int], node_states: Tensor) -> Tensor:
        # Loopless implementation, faster than the reference implementation.
        cumsum_indices = torch.tensor(batch_num_objects, device=self.get_device()).cumsum(0) - 1  # TODO: This can be computed once.
        cumsum_states = self.pre(node_states).cumsum(0).index_select(0, cumsum_indices)
        aggregated_states = torch.cat((cumsum_states[0].view(1, -1), cumsum_states[1:] - cumsum_states[0:-1]))
        return self.post(aggregated_states)
        # Reference implementation.
        # return self.post(torch.stack([torch.sum(nodes, dim=0) for nodes in self.pre(node_states).split(batch_num_objects)]))

    def feature_vectors(self, batch_num_objects: List[int], node_states: Tensor) -> Tensor:
        results: List[Tensor] = []
        offset: int = 0
        nodes: Tensor = self.pre(node_states)
        for num_objects in batch_num_objects:
            intermediate = []
            intermediate.append(torch.sum(nodes[offset:(offset + num_objects)], dim=0))
            for layer in self.post:
                intermediate.append(layer(intermediate[-1]))
            results.append(torch.cat(intermediate))
            offset += num_objects
        return torch.stack(results)


class RelationMessagePassingModel(nn.Module):
    def __init__(self, relations: list, hidden_size: int, iterations: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.iterations = iterations
        self.relation_network = RelationMessagePassing(relations, hidden_size)
        self.global_readout = Readout(hidden_size, hidden_size)
        self.readout_update = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size), nn.ReLU(), nn.Linear(2 * hidden_size, hidden_size))
        self.dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy.device

    def forward(self, states: Tuple[Dict[int, Tensor], List[int]]) -> Tensor:
        node_states = self._initialize_nodes(sum(states[1]))
        node_states = self._pass_messages(node_states, states[0], states[1])
        return node_states

    def _pass_messages(self, node_states: Tensor, relations: Dict[int, Tensor], batch_num_objects: List[int]) -> Tensor:
        for _ in range(self.iterations):
            node_states = self.relation_network(node_states, relations)
            readout = self.global_readout(batch_num_objects, node_states)
            readout_msg = torch.cat([readout[index].expand(num_objects, -1) for index, num_objects in enumerate(batch_num_objects)], dim=0)
            update_msg = torch.cat((node_states, readout_msg), dim=1)
            node_states = self.readout_update(update_msg)
        return node_states

    def _initialize_nodes(self, num_objects: int) -> Tensor:
        init_zeroes = torch.zeros((num_objects, (self.hidden_size // 2) + (self.hidden_size % 2)), dtype=torch.float, device=self.get_device())
        init_random = torch.randn((num_objects, self.hidden_size // 2), device=self.get_device())
        init_nodes = torch.cat([init_zeroes, init_random], dim=1)
        return init_nodes


class MaxReadoutModelBase(pl.LightningModule):
    def __init__(self, predicates: List[Tuple[str, int]], hidden_size: int, iterations: int):
        super().__init__()
        self.save_hyperparameters()
        encoding = dict([(predicate, index) for index, (predicate, _) in enumerate(predicates)])
        arities = [(encoding[predicate], arity) for predicate, arity in predicates]
        self.encoding = encoding
        self.model = RelationMessagePassingModel(arities, hidden_size, iterations)
        self.value_readout = Readout(hidden_size, 1)
        self.solvable_readout = Readout(hidden_size, 1)

    def forward(self, states: Tuple[Dict[str, Tensor], List[int]]) -> Tensor:
        encoded_states = (dict([(self.encoding[name], values) for name, values in states[0].items()]), states[1])
        node_states = self.model(encoded_states)
        value = torch.abs(self.value_readout(encoded_states[1], node_states))
        solvable = torch.sigmoid(self.solvable_readout(encoded_states[1], node_states))
        return value, solvable

    def feature_vectors(self, states: Tuple[Dict[int, Tensor], List[int]]) -> Tensor:
        encoded_states = (dict([(self.encoding[name], values) for name, values in states[0].items()]), states[1])
        node_states = self.model(encoded_states)
        value = self.value_readout.feature_vectors(encoded_states[1], node_states)
        solvable = self.solvable_readout.feature_vectors(encoded_states[1], node_states)
        return value, solvable

    def freeze_relation_model(self):
        """Freeze the relation message passing model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_relation_model(self):
        """Unfreeze the relation message passing model."""
        for param in self.model.parameters():
            param.requires_grad = True