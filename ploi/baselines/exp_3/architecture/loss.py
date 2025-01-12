import torch
import torch.nn as nn
from torch.functional import Tensor

g_suboptimal_factor = 2.0
g_loss_constants = [ 1.0, 1.0, 1.0, 1.0 ]

def set_suboptimal_factor(suboptimal_factor):
    global g_suboptimal_factor
    g_suboptimal_factor = suboptimal_factor

def set_loss_constants(loss_constants):
    global g_loss_constants
    g_loss_constants = loss_constants

def smax(x, k):
    return torch.div(torch.log(torch.sum(torch.exp(torch.mul(k, x)))), k)

def smin(x, k):
    return torch.mul(-1, smax(torch.mul(-1, x), k))

def l1_regularization(model: nn.Module, factor: float) -> Tensor:
    loss = 0.0
    if factor > 0.0:
        for parameter in model.parameters():
            loss += torch.sum(factor * torch.abs(parameter))
    return loss

def supervised_optimal_loss(output, target):
    values, solvables = output
    avg_abs_loss = torch.mean(torch.abs(torch.sub(target, values)))
    return avg_abs_loss
    # # TODO: Test if the following improves training.
    # # (Compares if two arbitrary states in the batch has the expected absolute value difference.)
    # output_pairs = torch.combinations(values.view(-1)).T
    # target_pairs = torch.combinations(target.view(-1)).T
    # output_diffs = output_pairs[0] - output_pairs[1]
    # target_diffs = target_pairs[0] - target_pairs[1]
    # consistency = torch.abs(output_diffs - target_diffs)
    # avg_consistency_loss = torch.mean(consistency)
    # return (avg_abs_loss + avg_consistency_loss) / 2.0

def unsupervised_optimal_loss(output, labels, solvable_labels, state_counts, device):
    loss = 0.0
    offset = 0
    values, solvables = output
    for index, state_count in enumerate(state_counts):
        value_prediction = values[offset][0]
        value_label = labels[index]
        if value_label == 0:
            loss += value_prediction
            offset += state_count
            assert state_count == 1
        else:
            # |V(s) - 1 + min_{s'} V(s')|
            successors = values[(offset + 1):(offset + 1 + (state_count - 1))].flatten()
            loss += torch.abs(value_prediction - (1.0 + torch.min(successors)))
            offset += state_count
    return loss / len(state_counts)


def selfsupervised_optimal_loss(output, labels, solvable_labels, state_counts, device):
    global g_suboptimal_factor
    loss = 0.0
    offset = 0
    values, solvables = output
    for index, state_count in enumerate(state_counts):
        value_prediction = values[offset][0]
        solvable_prediction = solvables[offset][0]
        value_label = labels[index]
        is_solvable = value_label < 2000000000
        solvable_label = torch.tensor(1.0 if is_solvable else 0.0, device=solvable_prediction.device)
        loss += torch.binary_cross_entropy_with_logits(solvable_prediction, solvable_label)
        if is_solvable:  # Is an solvable state, apply loss on value prediction
            if value_label == 0:
                loss += value_prediction
                assert state_count == 1
            else:
                # max(0, (1 + min_{s'} V(s')) - V(s) for all successor states s' of s
                successors = values[(offset + 1):(offset + 1 + (state_count - 1))].flatten()
                loss += torch.abs(value_prediction - (1.0 + torch.min(successors)))
                loss += torch.clamp(value_label - value_prediction, 0.0)
                loss += torch.clamp(value_prediction - g_suboptimal_factor * value_label, 0.0)
        offset += state_count
    return loss / len(state_counts)


def selfsupervised_suboptimal2_loss(output, head_labels, solvable_labels, state_counts, device):
    global g_suboptimal_factor
    assert len(head_labels) == len(state_counts)

    # read output from net
    output_values, output_solvables = output
    assert len(output_values) == len(output_solvables)

    # prepare data for calculating loss
    # batch consists of heads accompanied with their successors, number of heads is equal to len(state_counts)
    # head_labels contains true values for heads and solvable_labels contains solvability bit for each state

    #print(f'\n*** NEW LOSS ***')
    #print(f'state_counts={state_counts}')
    #print(f'head_labels={head_labels}')
    #print(f'solvable_labels={solvable_labels}')
    #print(f'output_values={output_values.flatten()}')
    #print(f'output_solvables={output_solvables.flatten()}')

    head_offsets = torch.cumsum(torch.cat([ torch.zeros(1, dtype=torch.int32, device=device), state_counts ]), dim=0)
    head_values = output_values[head_offsets[:-1]].flatten()
    head_solvables = solvable_labels[head_offsets[:-1]]
    goal_values = head_values[head_labels == 0]
    number_heads, number_goals, number_states = len(state_counts), len(goal_values), head_offsets[-1]
    #print(f'{number_heads} head(s): offsets={head_offsets[:-1]}, labels={head_labels}, values={head_values}, solvables={head_solvables}')
    #print(f'{number_goals} goal(s): values={goal_values}')

    # loss due to prediction of solvability
    _loss = g_loss_constants[0] * torch.sum(torch.binary_cross_entropy_with_logits(output_solvables.flatten(), solvable_labels.float())) / number_states
    xent_loss = 0.0 if g_loss_constants[0] == 0 else float(_loss / g_loss_constants[0])

    # main loss: V(s) >= 1 + min_{s'} V(s')          where min is over all *solvable* successor states s' of s
    #               0 >= (1 + min_{s'} V(s')) - V(s) where min is over all *solvable* successor states s' of s
    # main loss: max(0, (1 + min_{s'} V(s')) - V(s)) where min is over all *solvable* successor states s' of s
    solvable_and_non_goal_heads_mask = torch.logical_and(head_solvables, head_labels > 0)
    #print(f'solvable_and_non_goal_heads_mask={solvable_and_non_goal_heads_mask}')
    number_solvable_and_non_goal_heads = torch.sum(solvable_and_non_goal_heads_mask)
    main_loss = 0.0
    if number_solvable_and_non_goal_heads > 0:
        successors_values = [ output_values[(head_offsets[i]+1):head_offsets[i+1]].flatten() for i in range(number_heads) if solvable_and_non_goal_heads_mask[i] ]
        successors_solvables = [ solvable_labels[(head_offsets[i]+1):head_offsets[i+1]].flatten() for i in range(number_heads) if solvable_and_non_goal_heads_mask[i] ]
        alive_successors_values = [ torch.masked_select(z[0], z[1]) for z in zip(successors_values, successors_solvables) ]
        #print(f'successors_values={successors_values}')
        #print(f'successors_solvables={successors_solvables}')
        #print(f'alive_successors_values={alive_successors_values}')

        min_values = torch.cat([ torch.min(successors, 0, True)[0] for successors in alive_successors_values ])
        main_losses = torch.max(torch.stack([ torch.zeros_like(min_values, device=device), 1.0 + (min_values - head_values[solvable_and_non_goal_heads_mask]) ]), dim=0)[0]
        main_loss = torch.sum(main_losses) / number_solvable_and_non_goal_heads
        #print(f'main_losses={main_losses}, total={main_loss}')
    _loss += g_loss_constants[1] * main_loss

    # clamps penalize head_values that doesn't satisfy: label <= value <= g_suboptimal_factor * label
    # loss due to goal states not evaluating to zero captured by clamps
    number_solvable_heads = torch.sum(head_solvables)
    clamp1_loss, clamp2_loss = 0.0, 0.0
    if number_solvable_heads > 0:
        clamp1_loss = torch.sum(torch.clamp(head_labels[head_solvables] - head_values[head_solvables], min=0.0)) / number_solvable_heads
        clamp2_loss = torch.sum(torch.clamp(head_values[head_solvables] - g_suboptimal_factor * head_labels[head_solvables], min=0.0)) / number_solvable_heads
    _loss += g_loss_constants[2] * clamp1_loss + g_loss_constants[3] * clamp2_loss

    #print(f'selfsupervised_suboptimal2_loss: losses: total={_loss:7.4f}: xent={xent_loss:7.4f}, main={main_loss:7.4f}, clamp1={clamp1_loss:7.4f}, clamp2={clamp2_loss:7.4f}')
    return _loss

def selfsupervised_suboptimal_loss(output, labels, solvable_labels, state_counts, device):
    global g_suboptimal_factor
    loss = 0.0
    offset = 0
    values, solvables = output
    for index, state_count in enumerate(state_counts):
        value_prediction = values[offset][0]
        solvable_prediction = solvables[offset][0]
        value_label = labels[index]
        is_solvable = value_label < 2000000000
        solvable_label = torch.tensor(1.0 if is_solvable else 0.0, device=solvable_prediction.device)
        loss += torch.binary_cross_entropy_with_logits(solvable_prediction, solvable_label)
        if is_solvable:  # Is an solvable state, apply loss on value prediction
            if value_label == 0:
                loss += value_prediction
                assert state_count == 1
            else:
                # max(0, (1 + min_{s'} V(s')) - V(s) for all successor states s' of s
                successors = values[(offset + 1):(offset + 1 + (state_count - 1))].flatten()
                min_value_successor = torch.min(successors)
                loss += torch.max(torch.stack((torch.tensor(0.0, device=device), 1.0 + (min_value_successor - value_prediction))))
                loss += torch.clamp(value_label - value_prediction, 0.0)
                loss += torch.clamp(value_prediction - g_suboptimal_factor * value_label, 0.0)
        offset += state_count
    return loss / len(state_counts)

def unsupervised_suboptimal_loss(output, labels, state_counts, device):
    global g_suboptimal_factor
    loss = 0.0
    offset = 0
    values, solvables = output
    for index, state_count in enumerate(state_counts):
        value_prediction = values[offset][0]
        solvable_prediction = solvables[offset][0]
        value_label = labels[index]
        is_solvable = value_label < 2000000000
        solvable_label = torch.tensor(1.0 if is_solvable else 0.0, device=solvable_prediction.device)
        loss += torch.binary_cross_entropy_with_logits(solvable_prediction, solvable_label)
        if is_solvable:  # Is an solvable state, apply loss on value prediction
            if value_label == 0:
                loss += value_prediction
                assert state_count == 1
            else:
                # max(0, (1 + min_{s'} V(s')) - V(s) for all successor states s' of s
                successors = values[(offset + 1):(offset + 1 + (state_count - 1))].flatten()
                min_value_successor = torch.min(successors)
                loss += torch.max(torch.stack((torch.tensor(0.0, device=device), 1.0 + (min_value_successor - value_prediction))))
        offset += state_count
    return loss / len(state_counts)
