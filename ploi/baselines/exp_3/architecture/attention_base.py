import torch
import torch.nn as nn
import pytorch_lightning as pl

# Imports related to type annotations
from typing import List, Dict, Tuple
from torch.nn.functional import Tensor, hinge_embedding_loss


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
        self.query_weight = nn.Linear(hidden_size, hidden_size, False)
        self.key_weight = nn.Linear(hidden_size, hidden_size, False)
        self.value_weight = nn.Linear(hidden_size, hidden_size, False)
        self.update = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size, True), nn.ReLU(), nn.Linear(2 * hidden_size, hidden_size, True))
        self.dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy.device

    def forward(self, node_states: Tensor, relations: Dict[int, Tensor]) -> Tuple[Tensor, Tensor]:
        # Compute an aggregated message for each recipient
        messages = [[] for _ in range(node_states.shape[0])]
        for relation, module in enumerate(self.relation_modules):
            if (module is not None) and (relation in relations):
                values = relations[relation]
                input = torch.index_select(node_states, 0, values).view(-1, module[0].in_features)
                output = module(input).view(-1, self.hidden_size)
                node_indices = values.view(-1, 1)
                for index, node_index in enumerate(node_indices):
                    messages[node_index].append(output[index])
        queries = self.query_weight(node_states)
        lengths = [len(message) for message in messages]
        starts = [sum(lengths[:index]) for index in range(len(lengths))]
        ends = [starts[index] + lengths[index] for index in range(len(lengths))]
        messages = torch.cat([torch.stack(message) for message in messages])
        keys = self.key_weight(messages)
        values = self.value_weight(messages)
        attentions = torch.cat([torch.matmul(torch.softmax(torch.div(torch.matmul(queries[starts[index]:ends[index]], keys[starts[index]:ends[index]].T), lengths[index] ** 0.5), dim=0), values[starts[index]:ends[index]]) for index in range(len(lengths))]).squeeze()
        # attentions_2 = torch.matmul(torch.div(queries, keys.T), values)
        #  if lengths[index] > 0 else torch.zeros(self.hidden_size, device=self.get_device())

        # messages = [torch.stack(message) for message in messages]
        # keys = [self.key_weight(message).T for message in messages]
        # values = [self.value_weight(message) for message in messages]
        # attention_messages = torch.stack([torch.matmul(torch.softmax(torch.div(torch.matmul(queries[index], keys[index]), messages[index].shape[0] ** 0.5), dim=0), values[index]) for index in range(len(messages))]).squeeze()

        # Update states with aggregated messages
        next_node_states = self.update(torch.cat([attentions, node_states], dim=1))
        return next_node_states


class Readout(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(input_size, input_size, bias), nn.ReLU(), nn.Linear(input_size, input_size, bias))
        self.post = nn.Sequential(nn.Linear(input_size, input_size, bias), nn.ReLU(), nn.Linear(input_size, output_size, bias))

    def forward(self, batch_num_objects: List[int], node_states: Tensor) -> Tensor:
        results: List[Tensor] = []
        offset: int = 0
        nodes: Tensor = self.pre(node_states)
        for num_objects in batch_num_objects:
            results.append(self.post(torch.sum(nodes[offset:(offset + num_objects)], dim=0)))
            offset += num_objects
        return torch.stack(results)

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
        self.readout = Readout(hidden_size, 1)
        self.dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy.device

    def forward(self, states: Tuple[Dict[int, Tensor], List[int]]):
        node_states = self._initialize_nodes(sum(states[1]))
        node_states = self._pass_messages(node_states, states[0])
        return self.readout(states[1], node_states)

    def feature_vectors(self, states: Tuple[Dict[int, Tensor], List[int]]):
        node_states = self._initialize_nodes(sum(states[1]))
        node_states = self._pass_messages(node_states, states[0])
        return self.readout.feature_vectors(states[1], node_states)

    def _pass_messages(self, node_states: Tensor, relations: Dict[int, Tensor]) -> Tuple[Tensor, Tensor]:
        for _ in range(self.iterations):
             node_states = self.relation_network(node_states, relations)
        return node_states

    def _initialize_nodes(self, num_objects: int) -> Tensor:
        init_zeroes = torch.zeros((num_objects, (self.hidden_size // 2) + (self.hidden_size % 2)), dtype=torch.float, device=self.get_device())
        init_random = torch.randn((num_objects, self.hidden_size // 2), device=self.get_device())
        init_nodes = torch.cat([init_zeroes, init_random], dim=1)
        return init_nodes


class AttentionModelBase(pl.LightningModule):
    def __init__(self, predicates: List[Tuple[str, int]], hidden_size: int, iterations: int):
        super().__init__()
        self.save_hyperparameters()
        encoding = dict([(predicate, index) for index, (predicate, _) in enumerate(predicates)])
        arities = [(encoding[predicate], arity) for predicate, arity in predicates]
        self.encoding = encoding
        self.model = RelationMessagePassingModel(arities, hidden_size, iterations)

    def forward(self, states: Tuple[Dict[str, Tensor], List[int]]):
        encoded_states = (dict([(self.encoding[name], values) for name, values in states[0].items()]), states[1])
        return torch.abs(self.model(encoded_states))

    def feature_vectors(self, states: Tuple[Dict[str, Tensor], List[int]]):
        encoded_states = (dict([(self.encoding[name], values) for name, values in states[0].items()]), states[1])
        return torch.abs(self.model.feature_vectors(encoded_states))