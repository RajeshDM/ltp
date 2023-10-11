import copy
import warnings

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import  Data,HeteroData

import pddlgym
from pddlgym.structs import Predicate
from icecream import ic
import time
import os
import math
import pickle

import ploi.constants as constants


from .planning import PlanningFailure, PlanningTimeout

class TorchGraphDictDataset(Dataset):
    def __init__(self, graph_dicts_input, graph_dicts_target,pyg_inputs,pyg_outputs):
        self.graph_dicts_input = graph_dicts_input
        self.graph_dicts_target = graph_dicts_target
        self.pyg_input = pyg_inputs
        self.pyg_target = pyg_outputs

    def __len__(self):
        return len(self.graph_dicts_input)

    def __getitem__(self, idx):
        sample = {
            "graph_input": self.graph_dicts_input[idx],
            "graph_target": self.graph_dicts_target[idx],
            "pyg_input" : self.pyg_input[idx],
            "pyg_output" : self.pyg_target[idx],
        }
        return sample


def wrap_goal_literal(x):
    """Convert a state to a required input representation. """
    if isinstance(x, Predicate):
        return Predicate(
            "WANT" + x.name,
            x.arity,
            var_types=x.var_types,
            is_negative=x.is_negative,
            is_anti=x.is_anti,
        )
    new_predicate = wrap_goal_literal(x.predicate)
    return new_predicate(*x.variables)


def reverse_binary_literal(x):
    if isinstance(x, Predicate):
        assert x.arity == 2
        return Predicate(
            "REV" + x.name,
            x.arity,
            var_types=x.var_types,
            is_negative=x.is_negative,
            is_anti=x.is_anti,
        )
    new_predicate = reverse_binary_literal(x.predicate)
    variables = [v for v in x.variables]
    assert len(variables) == 2

def graph_to_pyg_data(graph):
    hetero_data = HeteroData()
    all_dtype = torch.float32
    gnn_processing_info = ['nodes','senders','receivers','edges','globals']
    hetero_data['node'].x = torch.tensor(graph['nodes'],dtype=all_dtype) 
    hetero_data['node', 'sends','node'].edge_index = torch.tensor(
                                np.array([graph['senders'],
                                graph['receivers']]),
                                dtype=torch.long)
    hetero_data['node','sends','node'].edge_attr = torch.tensor(graph['edges'],dtype=all_dtype)
    hetero_data['globals'].x = torch.tensor(graph['globals'],dtype=all_dtype)   

    for key,items in graph.items():
        if key == 'action_scores' :
            hetero_data['target_action_scores'].x = torch.tensor(copy.deepcopy(items),dtype=all_dtype)
            hetero_data[key].x = torch.tensor(copy.deepcopy(items),dtype=all_dtype)
            continue
        if key == 'action_object_scores' :
            hetero_data['target_action_object_scores'].x = torch.tensor(copy.deepcopy(items),dtype=all_dtype)
            hetero_data[key].x = torch.tensor(copy.deepcopy(items),dtype=all_dtype)
            continue

        if key not in gnn_processing_info :
            hetero_data[key].x = torch.tensor(copy.deepcopy(items),dtype=torch.long)
    '''
    '''
    return hetero_data

def graph_dataset_to_pyg_dataset(graphs):
    pyg_dataset= []
    for i in range(0,len(graphs)):
        pyg_dataset.append(graph_to_pyg_data(graphs[i]))

    return pyg_dataset

def _state_to_graph_ltp(state,action_space=None,all_groundings=None,
                    prev_actions=None,prev_state=None,test=False,
                    graph_metadata=None,goal_state=None):
    """Create a graph from a State
    """
    #ic (type(state))
    graph_building_start_time = time.time()
    _node_feature_to_index = graph_metadata['node_feature_to_index']
    _edge_feature_to_index = graph_metadata['edge_feature_to_index']
    _num_node_features = graph_metadata['num_node_features']
    _num_edge_features = graph_metadata['num_edge_features']

    #_model_version = graph_metadata['model_version']
    _all_predicates = graph_metadata['all_predicates']
    assert _node_feature_to_index is not None, "Must initialize first"
    G = wrap_goal_literal
    R = reverse_binary_literal
    all_objects = sorted(state.objects)
    if action_space != None :
        all_actions = [k for k,v in action_space.items()]
    all_agents = []
    #all_predicates = []
    #all_agents = ['agent_1']
    #all_global_nodes = ["global_1"]
    max_action_arity = 0
    for action in action_space:
        if action.arity > max_action_arity:
            max_action_arity = action.arity

    #literals = [literal for literal in state.literals if literal.predicate.arity != 0]
    #goal_literals = [ G(literal) for literal in list(state.goal.literals) if literal.predicate.arity != 0]
    #literals = list(state.literals)
    literals = [literal for literal in sorted(state.literals)]
    #goal_literals = [G(literal) for literal in sorted(state.goal.literals)]
    goal_literals = [G(literal) for literal in sorted(goal_state.literals)]
    #goal_literals_without_g = list(state.goal.literals)
    #all_literals = list(state.literals) + list(state.goal.literals)
    #all_literals_without_g = literals + goal_literals_without_g

    all_literals = literals + goal_literals
    #all_literals = literals + goal_literals_without_g
    #all_literals = literals + goal_literals_without_g
    #node_to_objects = dict(enumerate(all_agents + all_objects + _all_predicates + all_actions))

    all_predicates = all_literals[:]
    all_predicates = _all_predicates[:]


    node_to_objects = dict(enumerate(all_agents + all_objects + all_predicates + all_actions))
    if test == True:
        pass
        #ic (node_to_objects)
    objects_to_node = {v: k for k, v in node_to_objects.items()}
    #node_to_actions = dict(enumerate(all_actions))
    #actions_to_node = {v: k for k, v in node_to_actions.items()}
    #ic (all_objects)
    #ic (node_to_objects)
    num_objects = len(all_objects)
    num_actions = len(all_actions)
    num_agents = len(all_agents)
    #num_predicates = len(_all_predicates)
    #num_predicates = len(all_literals)
    num_predicates = len(all_predicates)
    #num_globals = len(all_global_nodes)
    #num_nodes = num_objects+ num_actions + num_agents
    num_nodes = num_objects+ num_actions + num_agents + num_predicates
    #num_nodes = num_objects + num_actions + num_agents + num_globals

    action_nodes = []
    action_edges = []
    agent_node = []
    agent_action_edges = []
    agent_object_edges = []
    agent_edges = []

    for action in all_actions :
        action_nodes.append(action)

    for action_node in action_nodes:
        for object in all_objects:
            action_edges.append((action_node,object))

    if all_agents != [] :
        for agent in all_agents:
            for object in all_actions :
                agent_action_edges.append((agent,object,0))
                agent_edges.append((agent,object,0))
            for object in all_objects :
                agent_object_edges.append((agent,object,1))
                agent_edges.append((agent, object, 1))
            #agent_edges

    edge_to_action_object = dict(enumerate(action_edges))
    #action_object_to_edge = {v: k for k, v in edge_to_action_object.items()}
    graph_input = {}

    # Nodes: one per object
    input_node_features = np.zeros((num_nodes, _num_node_features))
    num_nodes = np.array(num_nodes)
    #num_nodes = np.reshape(num_nodes,[1]).astype(np.int64)
    #graph_input["n_node"] = np.array(num_nodes)
    #graph_input['n_node'] = torch.from_numpy(np.reshape(num_nodes,[1])).float() #.astype(np.int64)
    graph_input['n_node'] = np.reshape(num_nodes,[1]) #.astype(np.int64)
    #input_node_features = np.zeros((num_nodes, self._num_node_features + 3))
    action_scores = np.zeros((1,num_actions))
    #action_object_scores = np.zeros((1, num_objects)) #TODO - need to think about how to make this variable
    action_object_scores = np.zeros((max_action_arity, num_objects)) #TODO - need to think about how to make this variable
    n_non_action_nodes = num_objects + num_agents + num_predicates
    prev_state_copy = None

    if prev_state != None :
        prev_state_copy = prev_state.copy()
        prev_state_copy['prev_graph'] = None
    #graph_input['prev_graph'] = prev_state_copy

    if all_agents != [] :
        for agent_index,agent in enumerate (all_agents) :
            type_index = _node_feature_to_index['agent']
            input_node_features[agent_index,type_index] = 1

    # Add features for types
    for obj_index, obj in enumerate(all_objects):
        var_type_index = _node_feature_to_index[obj.var_type]
        input_node_features[obj_index+num_agents, var_type_index] = 1
        if 'object_node' in _node_feature_to_index:
            type_index = _node_feature_to_index['object_node']
            input_node_features[obj_index+num_agents, type_index] = 1

    #ic (all_actions)
    for action_index, action in enumerate(all_actions):
        action_feature_pos = _node_feature_to_index[action]
        type_index = _node_feature_to_index['action_node']
        #input_node_features[action_index+len(all_objects), type_index] = 1
        input_node_features[action_index+n_non_action_nodes, type_index] = 1
        input_node_features[action_index+n_non_action_nodes,action_feature_pos] = 1

    #ic (all_literals)
    #ic (_node_feature_to_index)
    #ic (_all_predicates)
    #for lit_index, lit in enumerate(all_literals):
    #for lit_index, lit in enumerate(_all_predicates):
    #for# lit_index, lit in enumerate(all_literals_without_g):
    for lit_index, pred in enumerate(all_predicates):
        type_index = _node_feature_to_index['predicate_node']
        pred_index = objects_to_node[pred]
        #pred = lit.predicate
        predicate_feature_index = _node_feature_to_index[pred]
        input_node_features[pred_index, type_index] = 1
        input_node_features[pred_index, predicate_feature_index] = 1

        '''
        if lit in goal_literals and 'goal_pred' in _node_feature_to_index:
            goal_index = _node_feature_to_index['goal_pred']
            input_node_features[pred_index, goal_index] = 1
        '''

    all_edge_features_stack = np.zeros((max_action_arity,num_nodes, num_nodes,_num_edge_features))
    all_edge_features = all_edge_features_stack[0][:]

    '''
    All edge features of predicate object are being added in this function
    '''
    # self.add_all_predicate_edge_info_in_graph(state,objects_to_node,all_edge_features)
    _add_all_predicate_edge_info_in_graph(all_literals, objects_to_node, all_edge_features,_edge_feature_to_index)
    #_add_all_predicate_edge_info_in_graph(all_predicates, objects_to_node, all_edge_features,_edge_feature_to_index)

    node_to_only_actions = dict(enumerate(all_actions))
    action_positions = dict(enumerate(list(range(max_action_arity))))
    actions_to_node_groundings = {v: {val: [] for key, val in action_positions.items()} for k, v in node_to_only_actions.items()}

    #action_groundings = [op for op in pyperplan_task.operators if op.applicable(current_state)]
    for grounding in all_groundings:
        action = grounding.predicate
        objects = grounding.variables
        for object_pos, obj in enumerate(objects):
            actions_to_node_groundings[action][object_pos].append(obj)

    #for position in range(max_action_arity):
    for action_edge in action_edges :
        for position in range(max_action_arity):
            pred_index = _edge_feature_to_index['action_object']
            action_index = objects_to_node[action_edge[0]]
            obj_index = objects_to_node[action_edge[1]]
            #all_edge_features[action_index, obj_index, pred_index] = 1
            #TODO - Change this - currently using same features both ways
            #all_edge_features[obj_index, action_index, pred_index] = 1

            position_index = _edge_feature_to_index['action_param_pos_' + str(position)]
            all_edge_features_stack[position,action_index, obj_index, pred_index] = 1
            all_edge_features_stack[position,obj_index, action_index, pred_index] = 1
            all_edge_features_stack[position,action_index, obj_index, position_index] = 1
            all_edge_features_stack[position,obj_index, action_index, position_index] = 1

    for action, values in actions_to_node_groundings.items():
        action_index = objects_to_node[action]
        for position,objects in values.items():
            for object in objects :
                object_loc = all_objects.index(object)
                _get_precondition_satisfaction_position(action, state.literals, all_objects, all_edge_features_stack[position],
                                                            action_space, action_index, object_loc,
                                                            position,_edge_feature_to_index)

    for action_edge in action_edges :
        for position in range(max_action_arity):
            #pred_index = _edge_feature_to_index['action_object']
            action_index = objects_to_node[action_edge[0]]
            obj_index = objects_to_node[action_edge[1]]
            assert (all_edge_features_stack[position, action_index, obj_index]==
                all_edge_features_stack[position,obj_index, action_index]).all()
            if sum (all_edge_features_stack[position,action_index, obj_index]) == 2:
                all_edge_features_stack[position, action_index, obj_index] = [0] * _num_edge_features
                all_edge_features_stack[position, obj_index, action_index] = [0] * _num_edge_features
    # Organize into expected representation

    receivers, senders, edges = [], [], []
    sender_rec = []
    for all_edge_features in all_edge_features_stack :
        adjacency_mat = np.any(all_edge_features, axis=2)
        #ic (adjacency_mat.size)
        #ic (np.argwhere(adjacency_mat))
        for sender, receiver in np.argwhere(adjacency_mat):
            edge = all_edge_features[sender, receiver]
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)
            sender_rec.append((sender,receiver))

    n_edge = len(edges)
    edges = np.reshape(edges, [n_edge, _num_edge_features])
    receivers = np.reshape(receivers, [n_edge]).astype(np.int64)
    senders = np.reshape(senders, [n_edge]).astype(np.int64)
    n_edge = np.reshape(n_edge, [1]).astype(np.int64)
    num_actions = np.reshape(num_actions,[1]).astype(np.int64)
    num_objects = np.reshape(num_objects+num_agents,[1]).astype(np.int64)
    num_non_action_nodes = np.reshape(num_objects+num_agents+num_predicates,[1]).astype(np.int64)
    max_action_arity = np.reshape(max_action_arity,[1]).astype(np.int64)

    '''
    graph_input["nodes"] = torch.from_numpy(input_node_features).float()
    graph_input["receivers"] = torch.from_numpy(receivers).float()
    graph_input["senders"] = torch.from_numpy(senders).float()
    graph_input["n_edge"] = torch.from_numpy(n_edge).float()
    graph_input["edges"] = torch.from_numpy(edges).float()
    #graph_input["globals"] = torch.from_numpy(np.zeros(1,len(input_node_features[-1]))).float()#np.zeros((1,len(input_node_features[-1]))) 
    graph_input['action_scores'] = torch.from_numpy(action_scores).float()
    graph_input['action_object_scores'] = torch.from_numpy(action_object_scores).float()
    graph_input['n_parameters'] = torch.from_numpy(np.array(max_action_arity)).float()
    graph_input['n_action'] = torch.from_numpy(np.array(num_actions)).float()
    graph_input['n_object'] = torch.from_numpy(np.array(num_objects+num_agents)).float()
    graph_input['n_non_action_nodes'] = torch.from_numpy(np.array(num_objects + num_agents + num_predicates)).float()

    # Globals
    globals_array = np.zeros((1,len(input_node_features[-1])))
    graph_input["globals"] = torch.from_numpy(globals_array).float()
    '''
    graph_input['action_scores'] = action_scores
    graph_input['action_object_scores'] = action_object_scores
    graph_input['n_parameters'] = max_action_arity
    graph_input['n_action'] = num_actions
    graph_input['n_object'] = num_objects
    graph_input['n_non_action_nodes'] = num_non_action_nodes
    #graph_input['n_non_action_nodes'] = np.array(num_objects + num_agents + num_predicates)
    graph_input["nodes"] = input_node_features
    graph_input["receivers"] = receivers
    graph_input["senders"] = senders
    graph_input["n_edge"] = n_edge
    graph_input["edges"] = edges
    # Globals
    globals_array = np.zeros((1,len(input_node_features[-1])))
    graph_input["globals"] = globals_array

    return graph_input,node_to_objects


def _add_all_predicate_edge_info_in_graph(all_literals, objects_to_node, 
                                          all_edge_features,_edge_feature_to_index):
    # ic ("New unary literals start")
    # Add features for unary state literals
    G = wrap_goal_literal
    # for lit in state.literals:
    for lit in all_literals:
        # ic (lit.__dict__)
        if len(lit.variables) == 0:
            continue
        #pred_index = objects_to_node[lit]
        pred_index = objects_to_node[lit.predicate]
        # ic (lit)
        # ic (pred_index)
        # ic (lit_index)
        for i in range(len(lit.variables)):
            # self.add_predicate_info_in_edge_in_graph(lit, objects_to_node, all_edge_features, pred_index,lit_index, i)
            _add_predicate_edge_info_in_graph(lit, objects_to_node,
                                              all_edge_features, pred_index, i,_edge_feature_to_index)

def _add_predicate_info_in_edge_in_graph(lit,objects_to_node,all_edge_features,pred_index,lit_index,pos):
    obj_index = objects_to_node[lit.variables[pos]]
    all_edge_features[pred_index, obj_index, lit_index] = 1
    all_edge_features[obj_index, pred_index, lit_index] = 1

def _add_predicate_edge_info_in_graph(lit,objects_to_node,
                                      all_edge_features,pred_index,pos,_edge_feature_to_index):
    obj_index = objects_to_node[lit.variables[pos]]
    obj_pos_index = _edge_feature_to_index['pred_pos_' + str(pos)]
    all_edge_features[pred_index, obj_index, obj_pos_index] = 1
    all_edge_features[obj_index, pred_index, obj_pos_index] = 1

def _get_precondition_satisfaction_position(curr_action, all_literals, all_objects,all_edge_features,
                                            action_space, action_index, object_index,
                                            position,_edge_feature_to_index):
    precond_for_current_action_object_pos_var = []
    precond_for_current_action_object = []
    for action in action_space:
        if action.name == curr_action:
            #ic (curr_action)
            all_action_preconds = action_space[curr_action].preconds.literals
            param_name = action_space[curr_action].params[position]
            #ic (all_action_preconds)
            #ic (param_name)
            for precond in all_action_preconds:
                if len(precond.variables) == 0 :
                    precond_var = ''
                    precond_for_current_action_object_pos_var.append(precond_var)
                    precond_for_current_action_object.append(precond)
                for pos_in_precond, precond_var in enumerate(precond.variables) :
                    if param_name == precond_var :
                        precond_for_current_action_object_pos_var.append(precond_var)
                        precond_for_current_action_object.append(precond)

            for pos,precond in enumerate(precond_for_current_action_object):
                precond_var = precond_for_current_action_object_pos_var[pos]
                precond_str = str((precond.predicate))
                if len(precond.variables) == 0 :
                    curr_pos = 0
                else :
                    curr_pos = position + 1
                precondition_index = _edge_feature_to_index[precond_str+str(curr_pos)]
                all_edge_features[action_index, object_index, precondition_index] = 1
                # action_obj_inversion
                all_edge_features[object_index, action_index, precondition_index] = 1
            break


def _create_graph_dataset_ltp(training_data,dom_file=None,domain_name=None,agent=None,args=None):
    # Initializem the graph features

    # First get the types and predicates
    #self._no_params_predicates = set()
    _unary_types = set()
    _zero_size_predicate = set()
    _unary_predicates = set()
    _binary_predicates = set()
    _ternary_predicates = set()
    _agent_object_properties = ["Conf"]

    #action_space = training_data[4]
    action_space = training_data[3]
    if action_space != None:
        all_actions = [k for k, v in action_space.items()]
    #ic (training_data)

    for states in training_data[0]:
        for state in states :
            types = {o.var_type for o in state.objects}
            _unary_types.update(types)
            for lit in set(state.literals) | set(state.goal.literals):
                arity = lit.predicate.arity
                assert arity == len(lit.variables)
                #assert arity <= 2, "Arity > 2 predicates not yet supported"
                if arity == 0:
                    _zero_size_predicate.add(lit.predicate)
                elif arity == 1:
                    _unary_predicates.add(lit.predicate)
                elif arity == 2:
                    _binary_predicates.add(lit.predicate)
                elif arity >= 3 :
                    _ternary_predicates.add(lit.predicate)

    #ic (self._unary_predicates)
    #ic (self._binary_predicates)

    _unary_types = sorted(_unary_types)
    _zero_size_predicate = sorted(_zero_size_predicate)
    _unary_predicates = sorted(_unary_predicates)
    _binary_predicates = sorted(_binary_predicates)
    _ternary_predicates = sorted(_ternary_predicates)
    #ic (self._zero_size_predicate)
    #exit()
    _all_predicates = _zero_size_predicate + _unary_predicates + \
                        _binary_predicates + _ternary_predicates

    _num_predicates = len(_all_predicates)

    G = wrap_goal_literal
    R = reverse_binary_literal

    # Initialize node features
    _node_feature_to_index = {}
    index = 0
    _node_feature_to_index['action_node'] = index
    index += 1
    _node_feature_to_index['object_node'] = index
    index += 1
    _node_feature_to_index['predicate_node'] = index
    index += 1
    non_goal_predicates = []
    #non_goal_str_predicates = ['cfree','kin','motion']
    non_goal_str_predicates = ['cfree','kin','motion','trajcollision',
                                'CFreePosePose','CFreeApproachPose','CFreeTrajPose',
                                'FreeMotion','HoldingMotion','Traj',
                                'UnsafePose','UnsafeApproach','UnsafeTraj']
    non_goal_str_predicates = [elem.lower() for elem in non_goal_str_predicates]


    for action in all_actions:
        _node_feature_to_index[action] = index
        index += 1

    for predicate in _all_predicates:
        _node_feature_to_index[predicate] = index
        #self._node_feature_to_index[G(predicate)] = index
        index += 1
        #if predicate.name in non_goal_str_predicates:
        #    non_goal_predicates.append(predicate)

    for predicate in _all_predicates:
        if predicate in non_goal_str_predicates:
            continue
        _node_feature_to_index[G(predicate)] = index
        index += 1
    #ic (self._node_feature_to_index)
    #exit()
    '''
    '''

    #self._node_feature_to_index['prev_action'] = index
    #index += 1
    for unary_type in _unary_types:
        _node_feature_to_index[unary_type] = index
        index += 1

    cheating_input = args.cheating_input

    if cheating_input == True :
        _node_feature_to_index['is_correct_action'] = index
        index += 1
        _node_feature_to_index['is_correct_obj_1'] = index
        index += 1
        _node_feature_to_index['is_correct_obj_2'] = index
        index += 1

    #self._node_feature_to_index['goal_pred'] = index
    #index += 1

    '''
    for unary_predicate in self._unary_predicates:
        self._node_feature_to_index[unary_predicate] = index
        index += 1
    for unary_predicate in self._unary_predicates:
        self._node_feature_to_index[G(unary_predicate)] = index
        index += 1
    '''

    # Initialize edge features
    _edge_feature_to_index = {}
    index = 0
    #index += 1
    _edge_feature_to_index['action_object'] = index
    index += 1

    #TODO  adding diff edge type information - add it once basic learning starts working (sept 20, 2023)
    #_edge_feature_to_index['pred_object'] = index
    #index += 1

    '''
    Adding the preconditions in order instead of based on 
    each separate action
    '''
    #index_action_edge = index

    all_preconds = []

    #ic (action_space)
    #ic (action_space.__dict__)
    _num_action_edge_features = 0
    _max_objects = 0
    for key,value in action_space.items():
        #ic (key,value)
        #ic (value.__dict__)
        #required_features = self._num_action_edge_features + len(value.params)
        required_features = len(value.preconds.literals)
        #ic (required_features)
        #ic (value.__dict__)
        if _num_action_edge_features < required_features:
            _num_action_edge_features = required_features
        required_objects = len(value.params)
        if _max_objects < required_objects:
            _max_objects = required_objects
        #ic (value.preconds.literals)
        #ic (value.params)
        for precond in value.preconds.literals:
            #ic (precond.__dict__)
            predicate = str(precond.predicate)
            #objects_from_str = precond._str.split("(")[1:]
            s = precond._str
            objects_from_str = s[s.find("(")+1:s.find(")")].split(",")
            #ic (objects_from_str)
            positions = []
            for object in objects_from_str:
                if object == '' :
                    #ic (object)
                    positions.append(0)
                    break

                for param_pos, param in enumerate(value.params):
                    #ic (object)
                    if object == str(param):
                        positions.append(param_pos+1)

            #ic (positions)
            #if precond not in all_preconds :
            #    #ic (precond.__dict__)
            #    all_preconds.append(precond)
            for position in positions:
                if predicate + str(position) not in all_preconds:
                    all_preconds.append(predicate+str(position))

    #ic (all_preconds)
    '''
    for predicate in self._agent_object_properties:
        self._edge_feature_to_index[predicate] = index
        index += 1
    '''

    for arity in range(_max_objects):
        _edge_feature_to_index['action_param_pos_' + str(arity)] = index
        index+=1
    #self._action_edge_feature_to_index['action_object'] = index
    #exit()

    for precond in all_preconds:
        _edge_feature_to_index[precond] = index
        index+=1

    '''
    for precond in all_preconds:
        #self._action_edge_feature_to_index[precond._str] = index_action_edge
        #TODO - When there are predicates with more than 2 params this needs to be updated
        if len(precond.variables) == 0:
            if precond._str not in self._edge_feature_to_index:
                self._edge_feature_to_index[precond._str] = index
                index+=1
        for i,var in enumerate(precond.variables) :
            #ic (precond.__dict__)
            if str(precond.predicate) + str(i) not in self._edge_feature_to_index:
                self._edge_feature_to_index[str(precond.predicate) + str(i)] = index
                index+= 1
        if len(precond.variables) > 1:
            #self._action_edge_feature_to_index[precond._str + str(precond.variables[0])] = index_action_edge
            self._edge_feature_to_index[precond._str + str(precond.variables[0])] = index
            #index_action_edge += 1
            index += 1
            self._edge_feature_to_index[precond._str + str(precond.variables[1])] = index
            #self._action_edge_feature_to_index[precond._str + str(precond.variables[1])] = index_action_edge

        else :
            #self._action_edge_feature_to_index[precond._str] = index_action_edge
            self._edge_feature_to_index[precond._str] = index
        #index_action_edge += 1
        index += 1
        '''

    '''
    self._edge_feature_to_index['pos_x'] = index
    index+=1
    self._edge_feature_to_index['pos_y'] = index
    index+=1
    '''
    '''
    for i in range(0,self._max_objects):
        self._action_edge_feature_to_index['obj' +str(i)] = index_action_edge
        index_action_edge += 1
    '''

    '''
    Adding unary predicates as features of edges between
    object nodes and predicate nodes
    '''

    '''
    for unary_predicate in self._unary_predicates:
        self._edge_feature_to_index[unary_predicate] = index
        index += 1
    for unary_predicate in self._unary_predicates:
        self._edge_feature_to_index[G(unary_predicate)] = index
        index += 1

    for binary_predicate in self._binary_predicates:
        #ic (binary_predicate)
        #ic (type(binary_predicate))
        #ic (binary_predicate.__dict__)
        if binary_predicate.name in non_goal_str_predicates:
            non_goal_predicates.append(binary_predicate)

    for binary_predicate in self._binary_predicates:
        self._edge_feature_to_index[binary_predicate] = index
        index += 1
    for binary_predicate in self._binary_predicates:
        self._edge_feature_to_index[R(binary_predicate)] = index
        index += 1
    for binary_predicate in self._binary_predicates:
        if binary_predicate not in non_goal_predicates:
            self._edge_feature_to_index[G(binary_predicate)] = index
            index += 1
    for binary_predicate in self._binary_predicates:
        if binary_predicate not in non_goal_predicates:
            self._edge_feature_to_index[G(R(binary_predicate))] = index
            index += 1
    for ternary_predicate in self._ternary_predicates:
        self._edge_feature_to_index[ternary_predicate] = index
        index += 1
    '''
    _max_predicate_objects = 0
    for predicate in _all_predicates:
        #ic (predicate.arity)
        _max_predicate_objects = max(_max_predicate_objects,predicate.arity)

    '''
    if len (self._ternary_predicates) != 0 :
        self._max_predicate_objects = 3
    else :
        self._max_predicate_objects = 2
    '''

    for arity in range(_max_predicate_objects):
        _edge_feature_to_index['pred_pos_' + str(arity)] = index
        index+=1
    '''
    for ternary_predicate in self._ternary_predicates:
        for pred_index in len(ternary_predicate):
            self._edge_feature_to_index[str(ternary_predicate) + str()]
    '''

    _num_node_features = len(_node_feature_to_index)
    nnf = _num_node_features
    _num_edge_features= len(_edge_feature_to_index)
    nef = _num_edge_features

    # Process data
    num_training_examples = len(training_data[0])

    graphs_input = []
    graphs_target = []

    graph_metadata = {
        "num_node_features": _num_node_features,
        "num_edge_features": _num_edge_features,
        "node_feature_to_index": _node_feature_to_index,
        "edge_feature_to_index": _edge_feature_to_index,
        "unary_types": _unary_types,
        "unary_predicates": _unary_predicates,
        "binary_predicates": _binary_predicates,
        "all_predicates" : _all_predicates + [G(predicate) for predicate in _all_predicates]
        #"model_version" : model_version
    }
    if action_space != None:
        all_actions = [k for k, v in action_space.items()]
        num_actions =len(all_actions)

    for i in range(num_training_examples):
        #state = training_data[0][i]
        plan = training_data[2][i]
        all_graphs = []
        #ic (plan)
        for j,action in enumerate(plan) :
            state = training_data[0][i][j]
            #grounding = training_data[3][i][j]
            grounding = training_data[4][i][j]
            prev_actions = None
            if j == 1 :
                prev_actions = [plan[j-1]]
            curr_action, objects = _get_action_object_from_action(action)
            if j >= 1 :
                prev_state = all_graphs[j-1]
            else :
                prev_state = None
            prev_actions = None
            prev_state = None
            #graph_input,node_to_objects = self._state_to_graph_ltp(state,training_data[4],grounding,prev_actions,prev_state)

            if args.data_augmentation == True :
                goal_states = training_data[0][i][j+1:]
            else :
                goal_states = [state.goal] 

            for goal_state in goal_states :
                graph_input, graph_target = state_to_graph_wrapper(state,action_space,grounding,
                                                                prev_actions,prev_state,graph_metadata,
                                                                num_actions,curr_action,objects,goal_state,
                                                                cheating_input)
                all_graphs.append(graph_input)
                graphs_input.append(graph_input)
                graphs_target.append(graph_target)

    graphs_input,graphs_target = _expand_graph_to_max_size_features(graphs_input,graphs_target)
    return graphs_input,graphs_target, graph_metadata

def state_to_graph_wrapper(state,action_space,grounding,prev_actions,prev_state,graph_metadata,
                           num_actions,curr_action,objects,goal_state,cheating_input=False):
    graph_input,node_to_objects = _state_to_graph_ltp(state,action_space,grounding,prev_actions,prev_state,
                                                        graph_metadata=graph_metadata,goal_state=goal_state)

    # Target nodes
    literals = [literal for literal in sorted(state.literals)]
    G = wrap_goal_literal
    #goal_literals_old = [G(literal) for literal in sorted(state.goal.literals)]
    goal_literals = [G(literal) for literal in sorted(goal_state.literals)]
    num_objects = len(node_to_objects) - (num_actions) - (len(literals + goal_literals))
    num_non_action_nodes = len(node_to_objects) - num_actions
    objects_to_node = {v: k for k, v in node_to_objects.items()}
    action_scores = np.zeros((1,num_actions))
    action_index = objects_to_node[curr_action]
    action_scores[0][action_index - num_non_action_nodes] = 1
    max_number_action_parameters = 0
    for key,values in action_space.items():
        if len(values.params) > max_number_action_parameters:
            max_number_action_parameters = len(values.params)
    action_object_scores = np.zeros((max_number_action_parameters,num_objects))
    for o,curr_object in enumerate(objects):
        obj_index = objects_to_node[curr_object]
        action_object_scores[o][obj_index] = 1
    n_edge = graph_input['edges'].shape[0]
    max_number_action_parameters = np.array(max_number_action_parameters)
    n_edge = np.reshape(n_edge, [1]).astype(np.int64)

    graph_target = copy_info_from_graph(graph_input)

    add_extra_info_in_graph(graph_input, action_scores, num_actions,action_object_scores, objects, 
                            num_objects, max_number_action_parameters,action_index,cheating_input,objects_to_node)
    add_extra_info_in_graph(graph_target, action_scores, num_actions,action_object_scores, objects,
                                num_objects, max_number_action_parameters,action_index,cheating_input,objects_to_node)

    return graph_input, graph_target


def add_extra_info_in_graph(graph_input, action_scores, num_actions,action_object_scores,
                             objects, num_objects, max_number_action_parameters,
                             action_index,cheating_input,objects_to_node):
    graph_input['action_scores'] = np.reshape(np.array(action_scores),[1,num_actions]).astype(np.int64)
    graph_input['action_object_scores'] = np.reshape(np.array(action_object_scores),[max_number_action_parameters,num_objects]).astype(np.int64)
    graph_input['n_parameters'] = np.reshape(max_number_action_parameters,[1]).astype(np.int64)
    graph_input['target_n_parameters'] = np.reshape(np.array([len(objects)]),[1]).astype(np.int64)
    '''
    Adding in the cheating encoding to see if it overfits to just those features
    The Cheating encoding experiment 
    '''
    if cheating_input == True :
        graph_input['nodes'][action_index][-3] = 1
        for c,curr_object in enumerate(reversed(objects)):
            obj_index = objects_to_node[curr_object]
            graph_input['nodes'][obj_index][-1-c] =1

def copy_info_from_graph(graph_input):
    graph_target = {
        "n_node": graph_input["n_node"],
        "nodes" : graph_input["nodes"],
        "n_edge": graph_input["n_edge"],
        "edges": graph_input["edges"],
        "senders": graph_input["senders"],
        "receivers": graph_input["receivers"],
        "globals": graph_input["globals"],
        "action_scores": graph_input["action_scores"],
        "action_object_scores": graph_input["action_object_scores"],
        "n_action" : graph_input['n_action'],
        "n_object": graph_input['n_object'],
        #"prev_graph" : graph_input['prev_graph'],
        "n_non_action_nodes": graph_input['n_non_action_nodes']
    }
    return graph_target
    '''
    graph_target['nodes'] = object_mask
    graph_target['n_edge'] = n_edge
    graph_target['action_scores'] = np.reshape(np.array(action_scores),[1,num_actions]).astype(np.int64)
    graph_target['action_object_scores'] = np.reshape(np.array(action_object_scores),[max_number_action_parameters,num_objects]).astype(np.int64)
    graph_target['n_parameters'] = np.reshape(np.array([len(objects)]),[1]).astype(np.int64)
    graph_input['action_object_scores'] = np.reshape(np.array(action_object_scores),[max_number_action_parameters,num_objects]).astype(np.int64)
    graph_input['action_scores'] = np.reshape(np.array(action_scores),[1,num_actions]).astype(np.int64)
    graph_input['n_parameters'] = np.reshape(max_number_action_parameters,[1]).astype(np.int64)
    graph_input['target_n_parameters'] = np.reshape(np.array([len(objects)]),[1]).astype(np.int64)
    '''


def _expand_graph_to_max_size_features(graphs_input,graphs_target):
    max_action_object_score_size = 0
    for graph_input, graph_target in zip(graphs_input,graphs_target):
        input_shape = graph_input['action_object_scores'].shape[1]
        output_shape =  graph_target['action_object_scores'].shape[1]

        if max_action_object_score_size < input_shape :
            max_action_object_score_size = input_shape
        if max_action_object_score_size < output_shape :
            max_action_object_score_size = output_shape

    for graph_input, graph_target in zip(graphs_input, graphs_target):
        input_shape = graph_input['action_object_scores'].shape
        if input_shape[1] < max_action_object_score_size :
            result = np.zeros((input_shape[0],max_action_object_score_size))
            result[:input_shape[0],:input_shape[1]] = graph_input['action_object_scores']
            graph_input['action_object_scores'] = result

        input_shape = graph_target['action_object_scores'].shape
        if input_shape[1] < max_action_object_score_size :
            result = np.zeros((input_shape[0],max_action_object_score_size))
            result[:input_shape[0],:input_shape[1]] = graph_target['action_object_scores']
            graph_target['action_object_scores'] = result

    return graphs_input, graphs_target

def _get_action_object_from_action(action):
    return action.predicate, action.variables

def get_feasible_action_param_list(self,env,state,action_space,ensemble):
    groundings = env.action_space.all_ground_literals(state)
    action_param_list,_ = self.get_action_object_scores_ensemble(state,action_space._action_predicate_to_operators,groundings, prev_actions=None,
                                                                                prev_state=None, ensemble=ensemble)

    groundings = list(groundings)
    groundings_list = []
    possible_action_param_list = []
    for grounding in groundings:
        grounding_action = grounding.predicate
        objects = grounding.variables
        groundings_list.append(pddlgym.structs.Literal(grounding_action, objects))

    if len(groundings) == 0 :
        return [-1]
        #ic(action_param_list)
        #ic (state.literals)
        #ic (groundings)
        #ic (possible_action_param_list)
        #exit()
    for action_data in action_param_list :
        decoded_action, decoded_action_parameters = action_data[0], action_data[1]
        new_action = pddlgym.structs.Literal(decoded_action, decoded_action_parameters)
        in_grounding = False
        for grounded_action in groundings_list:
            in_grounding_temp = True
            if new_action.predicate == grounded_action.predicate:
                for grounded_var, action_var in zip(grounded_action.variables, new_action.variables):
                    if grounded_var != action_var:
                        in_grounding_temp = False
                        break
                if in_grounding_temp == True:
                    in_grounding = True
                    break
        if in_grounding == False :
            #number_impossible_actions += 1
            continue
        possible_action_param_list.append((new_action,action_data[2]))
        #state = env.step(new_action)
        #state = state[0]
        # continue
        #break
    return possible_action_param_list


def get_action_object_scores_ensemble( state, action_space,pyperplan_task ,
                                        prev_actions=None,prev_state=None
                                        ,correct_action_object_tuple=None,
                                        ensemble=False):

    #start_time = time.time()
    graph, node_to_objects = _state_to_graph_ltp(state,action_space,pyperplan_task,
                                                    prev_actions,prev_state,test=True,
                                                    graph_metadata=graph_metadata)
    object_idxs = []
    action_idxs = []
    action_param_tuples = []
    all_model_outputs = []
    '''
    for i,elem in enumerate(graph['nodes'][:-1]):
        if elem[0] == 1 :
            action_idxs.append(i)
        elif elem[0] == 0 :
            object_idxs.append(i)
    '''
    for i,elem in enumerate(graph['nodes']):
        if elem[0] == 1 :
            action_idxs.append(i)
        elif elem[0] == 0 :
            object_idxs.append(i)

    number_objects = len(object_idxs)

    if ensemble == True :
        for model in self._ensemble_models:
            predictions = _predict_multiple_graph_from_single_output_with_model(model,graph)
            output = _get_action_param_list_from_predictions(predictions,action_space,node_to_objects,number_objects)
            #ic (output)
            all_model_outputs.append(output)
    else :
        predictions = _predict_multiple_graph_from_single_output(graph)
        output = _get_action_param_list_from_predictions(predictions, action_space, node_to_objects, number_objects)
        all_model_outputs.append(output)

    all_possible_actions = {}

    for output in all_model_outputs:
        current_graph_keys = []
        for action_data in output :
            key = str(action_data[0]) + str(action_data[1])
            if key in current_graph_keys :
                continue
            current_graph_keys.append(key)
            if key not in all_possible_actions.keys():
                all_possible_actions[key] = [action_data,action_data[2]]
            else :
                current_score = all_possible_actions[key][1]
                all_possible_actions[key] = [action_data,current_score+action_data[2]]

    for key,value in all_possible_actions.items():
        action_param_tuples.append((value[0][0],value[0][1],value[1]))

    #output = sorted(action_param_tuples, key=lambda x: x[-1])
    #return output[::-1],graph
    return action_param_tuples,graph

def _get_action_param_list_from_predictions(predictions,action_space, node_to_objects,number_objects):

    action_param_tuples = []
    #ic (len(predictions))
    max_action_arity = 0
    for action in action_space:
        if action.arity > max_action_arity:
            max_action_arity = action.arity


    for prediction in predictions:
        obj_idxs = []
        action_idx = int(prediction[1][0][0])
        for arity in range(max_action_arity):
            obj_idxs.append(int(prediction[1][arity+1][arity]))
        tuple_score = prediction[3][0][0].item()
        action = node_to_objects[action_idx+number_objects]
        number_parameters = len(action_space[action].params)
        action_parameters = []
        tuple_object_score = 0
        divide_score_by = 1
        objects_score = 0
        for i in range(number_parameters):
            object = node_to_objects[obj_idxs[i]]
            action_parameters.append(object)
            objects_score += prediction[3][i+1][i].item()
            divide_score_by += 1
        action_param_tuples.append((action,action_parameters,tuple_score +objects_score/divide_score_by))

    return action_param_tuples

def _collect_training_data( train_env_name,load_existing_and_add_plans=False,collection_cycle=0,
                           outfile=None,_planner=None,_num_train_problems=None,args=None):
    """Returns X, Y where X are States and Y are sets of objects
    """
    #ic (train_env_name)
    _max_file_open = args.max_file_open
    ic (load_existing_and_add_plans)
    _load_dataset_from_file = True
    env = pddlgym.make("PDDLEnv{}-v0".format(train_env_name))
    use_existing_plans = False
    _debug_level = -1
    training_plan_lenghts = {}
    #outfile = "/"
    if not _load_dataset_from_file or not os.path.exists(outfile) or load_existing_and_add_plans:
        inputs = []
        outputs = []
        plans = []
        all_groundings = []
        action_space = env.action_space
        assert env.operators_as_actions
        plans_added_to_dataset = 0
        #for idx in range(min(self._num_train_problems, len(env.problems))):
        for idx in range(min(_max_file_open, _num_train_problems-(collection_cycle*_max_file_open))):
        #for idx in range(91,93):
        #for idx in range(300, 619):
            if load_existing_and_add_plans == True :
                curr_idx = (_max_file_open * collection_cycle) + idx
            else :
                curr_idx = idx

            ic (curr_idx)
            plan_file_loc = get_plan_file_loc(env,curr_idx)

            state_sequence = []
            if _debug_level == constants.max_debug_level -1 :
                print("Collecting training data problem {}".format(curr_idx),
                        flush=True)
            env.fix_problem_index(curr_idx)
            ic (env.problems[curr_idx].problem_fname)
            state, _ = env.reset()
            #ic (state)
            #for literal in state.literals:
            #    #ic (literal.__dict__)
            #    if literal.predicate.name == "motion" :
            #        #ic(literal.variables[0].__dict__)
            #        if literal.variables[0].name == 'q_1_0':
            #            ic (literal.__dict__)
            str_plan = False
            #ic (env.action_space.all_ground_literals(state))
            try:
                if os.path.exists(plan_file_loc) and use_existing_plans:
                    with open(plan_file_loc, "rb") as f:
                        plan = pickle.load(f)
                        #ic (plan)
                        str_plan = True
                else :
                    _planner.reset_statistics()
                    plan = _planner(env.domain, state, timeout=120)
                    #ic (plan)

            except (PlanningTimeout, PlanningFailure):
                print("Warning: planning failed, skipping: {}".format(
                    env.problems[curr_idx].problem_fname))
                continue
            #ic (plan[0])
            #ic (type(plan[0]))
            state_grounding = []

            if len(plan) in training_plan_lenghts:
                training_plan_lenghts[len(plan)] += 1
            else:
                training_plan_lenghts[len(plan)] = 1
            state_sequence.append(state)
            groundings = env.action_space.all_ground_literals(state)
            state_grounding.append(groundings)
            #state_grounding.append(env.action_space.all_ground_literals(state))
            new_plan = []
            #ic (groundings)
            #ic (plan)
            #continue

            in_grounding = True

            all_sub_plans = generate_repeated_sub_plans(env,plan,curr_idx)

            for action in plan:
                #ic (action)
                if str_plan == True :
                    action = convert_str_action_to_pddlgym_action(action,state_grounding[-1])
                    new_plan.append(action)
                #ic (action)
                new_state = env.step(action)
                state_sequence.append(new_state[0])
                groundings = env.action_space.all_ground_literals(new_state[0])
                #ic (groundings)
                #ic (len(groundings))
                if len(groundings) == 0:
                    #ic ("should be continuing:")
                    ic ("Grounding failure - data not collected")
                    in_grounding = False
                    #continue
                    break
                state_grounding.append(groundings)
                #state_grounding.append(env.action_space.all_ground_literals(new_state[0]))

            if in_grounding == False:
                continue
            #inputs.append(state)
            inputs.append(state_sequence)
            if str_plan == False:
                objects_in_plan = {o for act in plan for o in act.variables}
                plans.append(plan)
            else :
                #ic ("Writing new plan ")
                objects_in_plan = {o for act in new_plan for o in act.variables}
                plans.append(new_plan)

            outputs.append(objects_in_plan)
            all_groundings.append(state_grounding)
            plans_added_to_dataset += 1
        #training_data_curr = (inputs, outputs,plans,all_groundings,action_space._action_predicate_to_operators)
        training_data_curr = (inputs, outputs,plans,action_space._action_predicate_to_operators,all_groundings)

        #ic (training_plan_lenghts)
        #ic (plans_added_to_dataset)
        #ic (load_existing_and_add_plans)
        #exit()
        if load_existing_and_add_plans:
            with open(outfile, "rb") as f:
                training_data_from_file = pickle.load(f)
            #training_data = (training_data_curr[0]+training_data_from_file[0],
            #                 tra)
            training_data = [None] * len(training_data_curr)
            for i in range(0,len(training_data_curr)):
                if i != 3 :
                    training_data[i] = training_data_curr[i] + training_data_from_file[i]
                else :
                    training_data[i] = action_space._action_predicate_to_operators

            #training_data = [training_data_curr[i] + training_data_from_file[i] for i in range(0,len(training_data_curr)-1)]
            #training_data.append(action_space._action_predicate_to_operators)
            training_data = tuple(training_data)
            ic ("number of plans", len(training_data[2]))
        else :
            training_data = training_data_curr
        #exit()
        with open(outfile, "wb") as f:
            pickle.dump(training_data, f)

    with open(outfile, "rb") as f:
        training_data = pickle.load(f)

    return training_data,None,env.domain.domain_name
    #return training_data

def _collect_training_data_ltp(train_env_name,_planner,_num_train_problems,outfile,args):
    _max_file_open = args.max_file_open
    #outfile =_dataset_file_prefix + "_{}.pkl".format(train_env_name)
    if os.path.exists(outfile):
        with open(outfile, "rb") as f:
            training_data_from_file = pickle.load(f)
    else :
        training_data_from_file = ([],[],[],[],[])
    if len(training_data_from_file[2]) < _num_train_problems:
        if _num_train_problems > _max_file_open :
            #ic (math.ceil(self._num_train_problems/self._max_file_open))
            #starting = int(len(training_data_from_file[2])/self._max_file_open)
            starting = math.ceil(len(training_data_from_file[2])/_max_file_open)
            ending = math.ceil(_num_train_problems / _max_file_open)
            #ending = starting + 1
            for i in range(starting,ending ):
                ic (i)
                if i == 0 :
                    training_data,dom_file,domain_name = _collect_training_data(train_env_name,False,i,outfile,
                                                                                _planner,_num_train_problems,
                                                                                args)
                else :
                    training_data,dom_file,domain_name = _collect_training_data(train_env_name,True,i,outfile,
                                                                                _planner,_num_train_problems,
                                                                                args)
        else:
            training_data, dom_file, domain_name = _collect_training_data(train_env_name, False, 0,outfile,
                                                                                _planner,_num_train_problems,
                                                                                args)
    else :
        env = pddlgym.make("PDDLEnv{}-v0".format(train_env_name))
        #training_data = training_data_from_file
        training_data = [None] * len(training_data_from_file)
        domain_name = env.domain.domain_name
        for i in range(0, len(training_data_from_file)):
            if i != 3:
                training_data[i] = training_data_from_file[i][:_num_train_problems]
            else:
                training_data[i] = training_data_from_file[i]

    #ic("number of plans", len(training_data[2]))
    #exit()
    return training_data,None,domain_name

def generate_repeated_sub_plans(env,plan,curr_idx):
    env.fix_problem_index(curr_idx)
    #ic (env.problems[curr_idx].problem_fname)
    state, _ = env.reset()
    for action in plan:
        return 1

def convert_str_action_to_pddlgym_action(action_str,groundings):

    for grounding in groundings:
        #ic (grounding.__dict__)
        if grounding.predicate == action_str[0]:
            found = True
            for var_index,var in enumerate(grounding.variables) :
                #ic (str(var.split(":")[0]))
                #ic (action_str[1][var_index])
                if str(var.split(":")[0]) != action_str[1][var_index]:
                    found = False
                    break
            if found == True :
                #ic (grounding)
                return grounding
    return None


def get_plan_file_loc(env,curr_idx):
    location = env.problems[curr_idx].problem_fname.split("/")
    filename = location[-1]
    filename_components = filename.split("_")
    plan_file_name = "plan"
    plan_file_loc = ""
    for component in filename_components[1:]:
        plan_file_name += "_" + component
    for loc in location[:-1]:
        plan_file_loc += loc + "/"
    plan_file_loc += "plans/" + plan_file_name
    return plan_file_loc

def get_filenames(dataset_size,train_env_name,epochs,_model_version,
                       representation_size,_save_model_prefix,_seed,
                       args):
    #_model_version = args.model_version
    #message_strings = [['','orig_v1_r7']]
    gnn_rounds = args.gnn_rounds
    _num_epochs = args.epochs
    _debug_level = args.debug_level
    message_strings = []
    concept_loc = args.concept_loc
    #epoch_number = args.epoch_number
    #epoch_number = _epoch_number
    if args.server == True :
        #self.save_folder = os.path.join(os.path.dirname(__file__), "model/intermediate_models")
        save_folder = os.path.join(os.path.abspath('.'),'model')
        save_folder = os.path.join(save_folder,'intermediate_models')
    else :
        save_folder = "/tmp"
    round_loc = gnn_rounds - 4
    for major_concept in constants.major_concepts:
        message_strings_versions = []
        for version in constants.versions:
            message_strings_rounds = []
            for round in constants.rounds :
                message_strings_rounds.append(major_concept+"_" + version + "_" + round)
            message_strings_versions.append(message_strings_rounds)
        message_strings.append(message_strings_versions)

    message_string = message_strings[concept_loc-1][_model_version-1][round_loc-1] + "_" + \
                        str(representation_size)+"_d" + str(dataset_size)

    message_string += '_h' + str(args.n_heads)
    message_string += '_aug' + str(args.data_augmentation)
    message_string += '_ad' + str(args.attention_dropout)
    message_string += '_d' + str(args.dropout)
    message_string += '_lr' + str(args.lr)

    if args.weight_decay != 0 :
        message_string += "_wd" + str(args.weight_decay)

    if _debug_level < constants.max_debug_level :
        ic (message_string)
        ic (representation_size,gnn_rounds,dataset_size)
        ic (_save_model_prefix)
    if epochs == None or epochs == 0 :
        model_outfile = _save_model_prefix+"_{}_{}.pt".format(train_env_name, message_string)
    else :
        model_outfile = os.path.join(save_folder,str(train_env_name)+ "_seed"+ str(_seed) + "_model" \
                                      + str(epochs) + "_" + message_string + ".pt")

    return model_outfile,message_string,save_folder