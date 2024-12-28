import argparse
import re
import torch

from collections import deque
from datasets import collate_drop_label as collate
from datasets.supervised.optimal import ValueDataset, LimitedDataset, ExtendedDataset
from pathlib import Path
from torch.nn.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from typing import List, Tuple


def _load_datasets(path: Path, max_samples_per_value: int):
    files = list(path.glob('*states.txt'))
    decoded_datasets = [LimitedDataset(ValueDataset(d, decode=True), max_samples_per_value) for d in files]
    encoded_datasets = [LimitedDataset(ValueDataset(d, decode=False), max_samples_per_value) for d in files]
    predicates = decoded_datasets[0].predicates
    return (ExtendedDataset(decoded_datasets, 1), ExtendedDataset(encoded_datasets, 1), predicates)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formula', required=True, type=Path, help='Path to file with formulas')
    parser.add_argument('--data', required=True, type=Path, help='Path to directory of state values')
    parser.add_argument('--max_samples_per_value', default=100, type=int, help='Maximum number of states per value per dataset')
    parser.add_argument('--model', required=True, type=Path, help='Path to the model')
    parser.add_argument('--model_type', required=True, type=str, help='Model type (add, max, addmax)')
    parser.add_argument('--mode', required=True, type=str, help='Train a transformation or evaluate an existing one on a dataset (train or test)')
    parser.add_argument('--path', required=True, type=Path, help='Model output path')
    args = parser.parse_args()
    return args


def _predicate(predicate, variables, first, second, invert):
    def evaluation(state: dict, context: dict):
        if predicate in state:
            objects = state[predicate]
            if invert: objects = [(y, x) for (x, y) in objects]
            if first: objects = [(x,) for x, _ in objects]
            if second: objects = [(y,) for _, y in objects]
            values = set(objects)
            arguments = tuple([context['scope'][variable] for variable in variables])
            return arguments in values
        return False
    return evaluation


def _find_corresponding_predicate(name: str, predicates):
    tokens = name.lower().split(':')
    first = False
    second = False
    filter = None
    invert = False
    for predicate, arity in predicates:
        if predicate.lower() == tokens[-1]:
            for token in tokens[:-1]:
                first |= token.startswith('first')
                second |= token.startswith('second')
                invert |= token.startswith('invert')
                if '|' in token:
                    filter, _, _, _, _, _ = _find_corresponding_predicate(token.split('|')[-1], predicates)
            return (predicate, arity, first, second, filter, invert)
    return (None, None, None, None, None, None)


def _and(first_body, second_body):
    return lambda state, context: first_body(state, context) and second_body(state, context)


def _exists(body, variable):
    def evaluation(state: dict, context: dict):
        previous_value = context['scope'][variable] if variable in context['scope'] else None
        result = False
        objects = context['objects']
        for obj in objects:
            context['scope'][variable] = obj
            result = body(state, context)
            if result: break
        # Restore previous value, if any.
        if previous_value: context['scope'][variable] = previous_value
        else: del context['scope'][variable]
        return result
    return evaluation


def _not(body):
    return lambda state, context: not body(state, context)


def _shortest_path(goal_body: str, relation_description: str, aux, variable, max_aux: str, predicates):
    (relation, _, _, _, _, relation_invert) = _find_corresponding_predicate(relation_description, predicates)
    def evaluation(state, context):
        obj = context['scope'][variable]

        if relation not in state:
            context['cache'][aux] = 0
            return goal_body(state, context)

        edges = {}
        relations = state[relation]
        if relation_invert: relations = [(y, x) for (x, y) in relations]
        for lhs, rhs in relations:
            edges.setdefault(lhs, []).append(rhs)
        max_depth = context['cache'][max_aux] if max_aux in context['cache'] else 10000
        previous_value = context['scope'][variable] if variable in context['scope'] else None
        found_depth = None

        queue = deque([(obj, 0)])
        closed = set()
        while len(queue) > 0:
            (x, d) = queue.popleft()
            if (d <= max_depth) and (x not in closed):
                closed.add(x)
                context['scope'][variable] = x
                if goal_body(state, context):
                    found_depth = d
                    break
                if x in edges:
                    for y in edges[x]:
                        queue.append((y, d + 1))

        # Restore previous value, if any.
        if previous_value: context['scope'][variable] = previous_value
        else: del context['scope'][variable]

        if found_depth is not None:
            context['cache'][aux] = found_depth
            return True
        else:
            context['cache'][aux] = 0
            return False
    return evaluation


def _multiplication(first_body, second_body):
    def evaluation(state, context):
        lhs = first_body(state, context)
        if abs(lhs) < 0.01:
            return lhs
        return lhs * second_body(state, context)
    return evaluation


def _addition(first_body, second_body):
    return lambda state, context: first_body(state, context) + second_body(state, context)


def _subtraction(first_body, second_body):
    return lambda state, context: first_body(state, context) - second_body(state, context)


def _constant_or_feature(feature: str):
    def evaluation(state, context):
        if feature.isdigit(): return int(feature)
        features = context['features']
        cache = context['cache']
        # Cache evaluations, some features might be used in multiple places.
        if feature in cache:
            result = cache[feature]
            if isinstance(result, bool):
                result = 1 if result else 0
            return result
        if feature in features: result = 1 if features[feature](state, context) else 0
        elif feature in features: result = 1 if features[feature](state, context) else 0
        else: raise Exception('Error: {} is not defined. Always place variables last in the term.'.format(feature))
        cache[feature] = result
        return result
    return evaluation


DEBUG = False
DEBUG_INDENT = 0


def _parse_feature_expression(index, expression, predicates: List[Tuple[str, int]], boolean_features: dict):
    operator = expression[index]

    global DEBUG
    global DEBUG_INDENT
    if 'DEBUG' in globals() and DEBUG:
        print('{}{}'.format(' ' * (2 * DEBUG_INDENT), operator))
    DEBUG_INDENT += 1

    if operator == 'NOT':
        index, body = _parse_feature_expression(index + 1, expression, predicates, boolean_features)
        DEBUG_INDENT -= 1
        return index, _not(body)
    elif operator == 'EXISTS':
        variable = expression[index + 1]
        index, body = _parse_feature_expression(index + 2, expression, predicates, boolean_features)
        DEBUG_INDENT -= 1
        return index, _exists(body, variable)
    elif operator == 'AND':
        index, first_body = _parse_feature_expression(index + 1, expression, predicates, boolean_features)
        index, second_body = _parse_feature_expression(index, expression, predicates, boolean_features)
        DEBUG_INDENT -= 1
        return index, _and(first_body, second_body)
    elif operator == 'SHORTEST_PATH':
        index, goal_body = _parse_feature_expression(index + 1, expression, predicates, boolean_features)
        relation = expression[index]
        aux = expression[index + 1]
        max_aux = expression[index + 2]
        variable = expression[index + 3]
        DEBUG_INDENT -= 1
        return index + 4, _shortest_path(goal_body, relation, aux, variable, max_aux, predicates)
    elif operator in boolean_features:
        DEBUG_INDENT -= 1
        return index + 1, boolean_features[operator]
    else:
        (predicate, arity, first, second, filter, invert) = _find_corresponding_predicate(operator, predicates)
        if first: arity = 1
        if second: arity = 1
        if filter: raise NotImplementedError()
        if predicate is not None:
            variables = expression[index + 1: index + arity + 1]
            DEBUG_INDENT -= 1
            return index + arity + 1, _predicate(predicate, variables, first, second, invert)
        raise Exception('Not a predicate: {}'.format(operator))


def _parse_value_expression(index, expression):
    operator = expression[index]
    if operator == '*':
        index, first_body = _parse_value_expression(index + 1, expression)
        index, second_body = _parse_value_expression(index, expression)
        return index, _multiplication(first_body, second_body)
    elif operator == '+':
        index, first_body = _parse_value_expression(index + 1, expression)
        index, second_body = _parse_value_expression(index, expression)
        return index, _addition(first_body, second_body)
    elif operator == '-':
        index, first_body = _parse_value_expression(index + 1, expression)
        index, second_body = _parse_value_expression(index, expression)
        return index, _subtraction(first_body, second_body)
    else:
        return index + 1, _constant_or_feature(operator)


def _parse_formula(file: Path, predicates: list):
    features = {}
    with file.open('r') as stream:
        lines = stream.readlines()
    for line in lines:
        if len(line.strip()) <= 0: continue
        lhs, rhs = line.split('=')
        lhs = lhs.strip()
        rhs = rhs.strip()
        rhs = rhs.replace('\n', '')
        rhs = [token for token in re.split(' |,|\(|\)', rhs) if len(token) > 0]
        if lhs.startswith('FEATURE') :
            name = lhs.strip().split(' ')[1]
            features[name] = _parse_feature_expression(0, rhs, predicates, features)[1]
        elif lhs.startswith('VALUE'):
            value_function = _parse_value_expression(0, rhs)[1]
        elif lhs.startswith('VECTOR'):
            if 'PREREQUISITES' in rhs:
                idx = rhs.index('PREREQUISITES')
                vector_tokens = rhs[0:idx]
                prereq_tokens = rhs[idx + 1:]
            else:
                vector_tokens = rhs
                prereq_tokens = []
            def vector_function_closure(state, context):
                for token in prereq_tokens:
                    features[token](state, context)
                feature_vector = []
                index = 0
                num_tokens = len(vector_tokens)
                while index < num_tokens:
                    token = vector_tokens[index]
                    feature_value = 0.0
                    if token == '+':
                        lhs_token = vector_tokens[index + 1]
                        rhs_token = vector_tokens[index + 2]
                        if lhs_token in features: feature_value += 1.0 if features[lhs_token](state, context) else 0.0
                        elif lhs_token in context['cache']: feature_value += context['cache'][lhs_token]
                        if rhs_token in features: feature_value += 1.0 if features[rhs_token](state, context) else 0.0
                        elif rhs_token in context['cache']: feature_value += context['cache'][rhs_token]
                        index += 2
                    elif token in features: feature_value = 1.0 if features[token](state, context) else 0.0
                    elif token in context['cache']: feature_value = context['cache'][token]
                    feature_vector.append(feature_value)
                    index += 1
                return feature_vector
            vector_function = vector_function_closure
        else:
            raise Exception('Format error')
    return (features, value_function, vector_function)


def _load_model(model_path: Path, model_type: str):
    from architecture import MaxModelBase as Model
    model = Model.load_from_checkpoint(checkpoint_path=str(model_path), strict=False)
    return model


def _label_data(model, encoded_dataset, decoded_dataset, value_function, vector_function, device):
    inputs = []
    labels = []
    # Generate input using model
    model.eval()
    with torch.no_grad():
        data_loader = DataLoader(encoded_dataset, batch_size=32, collate_fn=collate)
        for input_states in data_loader:
            for key, value in input_states[0].items():
                input_states[0][key] = value.to(device, non_blocking=True)
            input_vector = model.feature_vectors(input_states)
            for index in range(input_vector.shape[0]):
                inputs.append(input_vector[index])
    # Label using formal features
    for state, label in decoded_dataset:
            context = {
                'objects': set([obj for arguments in [arguments for argument_instance in state.values() for arguments in argument_instance] for obj in arguments]),
                'features': features,
                'scope': {},
                'cache': {}
            }
            label_vector = torch.tensor(vector_function(state, context)).to(device)
            labels.append(label_vector)
            label_prediction = value_function(state, context)
            assert int(label) == label_prediction
    return list(zip(inputs, labels))


def _loss(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).abs().sum(dim=1).mean()


def _train_step(transformation, optimizer, data_loader):
    total_loss = 0.0
    for (input_batch, target_batch) in data_loader:
        optimizer.zero_grad()
        pred_batch = transformation(input_batch)
        loss = _loss(pred_batch, target_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(data_loader)


def _train(transformation, data_loader):
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.00001)
    epochs = 100000
    for epoch in range(epochs):
        avg_loss = _train_step(transformation, optimizer, data_loader)
        if (epoch % 100) == 0:
            print('Epoch {} of {}: {:.5f}'.format(epoch + 1, epochs, avg_loss), flush=True)
    print('Train loss: {:.5f}'.format(avg_loss), flush=True)
    torch.save(transformation, args.path)


def _test(transformation, data_loader):
    transformation.eval()
    with torch.no_grad():
        final_loss = 0.0
        target_distribution = []
        error_distribution = []
        for (input_batch, target_batch) in data_loader:
            pred_batch = transformation(input_batch)
            loss = _loss(pred_batch, target_batch)
            target_distribution.append(target_batch)
            error_distribution.append((pred_batch - target_batch).abs())
            final_loss += loss
        print('Test loss: {}'.format(final_loss / len(data_loader)))
        error_distribution = torch.cat(error_distribution)
        error_mean = error_distribution.T.mean(dim=1)
        error_std = error_distribution.T.std(dim=1)
        target_distribution = torch.cat(target_distribution)
        target_mean = target_distribution.T.mean(dim=1)
        target_std = target_distribution.T.std(dim=1)
        list_error_mean = ["{:.2f}".format(x) for x in error_mean]
        list_target_mean = ["{:.2f}".format(x) for x in target_mean]
        list_error_std = ["{:.2f}".format(x) for x in error_std]
        list_target_std = ["{:.2f}".format(x) for x in target_std]
        str_error_mean = "Error Avg. = " + str.join(", ", list_error_mean)
        str_error_std = "Error Std. = " + str.join(", ", list_error_std)
        str_target_mean = "Target Avg. = " + str.join(", ", list_target_mean)
        str_target_std = "Target Std. = " + str.join(", ", list_target_std)
        foo = zip(list_error_mean, list_error_std)
        bar = zip(list_target_mean, list_target_std)
        str_foo = str.join(" & ", ["{} ({})".format(a, b) for a, b in foo])
        str_bar = str.join(" & ", ["{} ({})".format(a, b) for a, b in bar])
        print(str_error_mean)
        print(str_error_std)
        print(str_target_mean)
        print(str_target_std)
        print(str_foo)
        print(str_bar)


if __name__ == "__main__":
    args = _parse_arguments()
    print ('Loading model...', flush=True)
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    model = _load_model(args.model, args.model_type).to(device)
    print('Loading datasets...', flush=True)
    decoded_dataset, encoded_dataset, predicates = _load_datasets(args.data, args.max_samples_per_value)
    print('Parsing formula...', flush=True)
    (features, value_function, vector_function) = _parse_formula(args.formula, predicates)
    print('Evaluating...', flush=True)
    feature_data = _label_data(model, encoded_dataset, decoded_dataset, value_function, vector_function, device)
    print('Labeled {} states'.format(len(feature_data)))
    if args.path.exists():
        transformation = torch.load(args.path).to(device)
    else:
        if args.mode.lower() == 'test':
            raise Exception('Transformation must exist in test mode')
        in_features = feature_data[0][0].shape[0]
        out_features = feature_data[0][1].shape[0]
        transformation = torch.nn.Linear(in_features, out_features, True).to(device)
    data_loader = DataLoader(feature_data, batch_size=64, shuffle=True)
    if args.mode.lower() == 'train':
        _train(transformation, data_loader)
    _test(transformation, data_loader)