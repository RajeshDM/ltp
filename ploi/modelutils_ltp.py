import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from icecream import ic
from torch_geometric.nn import GATv2Conv,BatchNorm
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import aggr
from ploi.attention_layer import (
 GraphAttentionV2Layer,
 MLP,
)
from ploi.scriptable_models import HeteroGNN_global_Wrapper
import time
import itertools
from typing import Optional, Union,Tuple, Dict, Any, List
from torch_geometric.data import  Data,HeteroData

class EdgeModelLtp(nn.Module):
    def __init__(self, n_features, n_edge_features, n_global, n_hidden, dropout=0.0):
        #super(EdgeModelLtp, self).__init__()
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_features + n_edge_features + n_global, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )

    #def forward(self, src, dest, edge_attr, u=None, batch=None):
    def forward(self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor, 
                u: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u is not None:
            out = torch.cat([src,dest, edge_attr,u], dim=1)
        else :
            out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)

class EdgeModelLtp_2(nn.Module):
    def __init__(self, n_features, n_edge_features, n_global, n_hidden,num_mlp_layers , dropout=0.0):
        super().__init__()
        self.edge_mlp = MLP([n_hidden]*num_mlp_layers, 2*n_features + n_edge_features + n_global, dropout=dropout) 

    def forward(self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor, 
                u: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u is not None:
            out = torch.cat([src,dest, edge_attr,u], dim=1)
        else :
            out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)

class GlobalModel(nn.Module):
    def __init__(self, n_global_features,n_hidden, dropout=0.0):
        #super(GlobalModel, self).__init__()
        super().__init__()
        self.global_mlp_2 = nn.Sequential(
            #nn.Linear(n_hidden*3, n_hidden),
            nn.Linear(n_global_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )
        self.aggr = aggr.SumAggregation()

    #def forward(self, x, edge_index,edge_attr, u ,node_index):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
            edge_attr: torch.Tensor, u: torch.Tensor, 
            node_index: torch.Tensor) -> torch.Tensor:
        nodes_agg = self.aggr(x,index=node_index)
        edges_agg =self.aggr(edge_attr,index=edge_index) 

        u = torch.cat([u, nodes_agg,edges_agg],dim=1)
        return self.global_mlp_2(u)

class GlobalModel_v2(nn.Module):
    def __init__(self, n_global_features, n_hidden,num_mlp_layers,dropout=0.0):
        super().__init__()
        self.global_mlp_2 = MLP([n_hidden]*num_mlp_layers, n_global_features, dropout=dropout) 
        self.aggr = aggr.SumAggregation()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
            edge_attr: torch.Tensor, u: torch.Tensor, 
            node_index: torch.Tensor) -> torch.Tensor:
        nodes_agg = self.aggr(x,index=node_index)
        edges_agg =self.aggr(edge_attr,index=edge_index) 

        u = torch.cat([u, nodes_agg,edges_agg],dim=1)
        return self.global_mlp_2(u)


#Hetero graph neural network
class HeteroGNN_global(nn.Module):
    def __init__(self,n_features,n_edge_features,n_global_features,
                        representation_size ,
                        dropout=0.0,
                        attn_dropout=0.0,
                        num_rounds=3,
                        n_heads=1,
                        num_mlp_layers=2,
                        device='cuda:0'):
        super(HeteroGNN_global, self).__init__()
        
        num_mlp_layers_encoder = num_mlp_layers
        num_mlp_layers_update = num_mlp_layers_encoder 
        num_mlp_layers_update_global = num_mlp_layers_update
        self.device = device
        self.num_rounds = num_rounds
        self.representation_size = representation_size

        # Node related layers
        self.node_encoder = MLP([self.representation_size]*num_mlp_layers_encoder,n_features)
        self.node_attention_layer = GraphAttentionV2Layer(in_features_1=self.representation_size*2,
                                                    out_features_1=self.representation_size*2,
                                                    in_features_2=self.representation_size,
                                                    out_features_2=self.representation_size,
                                                    n_heads=n_heads,
                                                    is_concat=False,
                                                    dropout=attn_dropout,
                                                    leaky_relu_negative_slope=0.2,
                                                    share_weights=False)
        
        self.node_update = MLP([self.representation_size]*num_mlp_layers_update, self.representation_size*5,dropout=dropout)
        
        # Edge related layers
        self.edge_encoder = MLP([self.representation_size]*num_mlp_layers_encoder, n_edge_features)
        #self.edge_update = EdgeModelLtp_2(self.representation_size*2, #Node features
        self.edge_update = EdgeModelLtp(self.representation_size*2, #Node features
                                        self.representation_size*2, #Edge features
                                        self.representation_size*2, #global features
                                        self.representation_size, #hidden features
                                        #num_mlp_layers=num_mlp_layers_update,
                                        dropout=dropout)

        # Global related layers
        self.global_encoder  = MLP([self.representation_size]*num_mlp_layers_encoder,n_global_features)
        #self.global_update = GlobalModel_v2(self.representation_size*4,
        self.global_update = GlobalModel(self.representation_size*4,
                                         self.representation_size,
                                         #num_mlp_layers_update_global,
                                         dropout=dropout
                                         )
        
    #def forward(self, batch):
    def forward(self, batch:  Union[HeteroData, Dict[str, Any]])-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_data  = batch['node'].x
        node_index = batch['node'].batch
        #node_data  = batch['node']['x']
        #node_index = batch['node']['batch']
        #edge_index = torch.repeat_interleave(torch.arange(0, len(batch['globals'].x)).cuda(), batch['n_edge'].x)
        edge_index = torch.repeat_interleave(torch.arange(0, len(batch['globals'].x),device=self.device), batch['n_edge'].x)

        edge_features_node = batch['node','sends','node'].edge_attr
        edge_features_node_index = batch['node','sends','node'].edge_index
        #global_data = batch['globals']['x']
        global_data = batch['globals'].x

        src_indices  = edge_features_node_index[0]
        dest_indices = edge_features_node_index[1]
        
        node_data = self.node_encoder(node_data)
        edge_features_node = self.edge_encoder(edge_features_node)
        global_data = self.global_encoder(global_data)

        node_data_original = node_data
        edge_features_node_original = edge_features_node
        global_data_original = global_data

        for i in range(self.num_rounds) :
            node_data = torch.cat([node_data_original,node_data],dim=1)
            edge_features_node = torch.cat([edge_features_node_original,edge_features_node],dim=1)
            global_data = torch.cat([global_data_original,global_data],dim=1)

            src = node_data[src_indices]
            dest = node_data[dest_indices]
            global_node_repeat = global_data[node_index]
            global_edge_repeat = global_data[edge_index]

            edge_features_node = self.edge_update(src,dest,edge_features_node,global_edge_repeat)
            node_data = self.node_update(self.node_attention_layer(node_data,edge_features_node,dest_indices,global_node_repeat))
            global_data = self.global_update(node_data,edge_index,edge_features_node,global_data,node_index)
        return node_data, edge_features_node,global_data

class HeteroGNN(nn.Module):
    def __init__(self,n_features,n_edge_features,n_global_features,
                        representation_size ,
                        dropout=0.0,
                        attn_dropout=0.0,
                        num_rounds=3,
                        n_heads=1,
                        device='cuda:0'):
        super(HeteroGNN, self).__init__()
        
        self.device = device
        self.num_rounds = num_rounds
        self.representation_size = representation_size

        # Node related layers
        self.node_encoder = MLP([self.representation_size]*2,n_features)
        self.node_attention_layer = GraphAttentionV2Layer(in_features_1=self.representation_size*2,
                                                    out_features_1=self.representation_size*2,
                                                    in_features_2=self.representation_size,
                                                    out_features_2=self.representation_size,
                                                    n_heads=n_heads,
                                                    is_concat=False,
                                                    dropout=attn_dropout,
                                                    leaky_relu_negative_slope=0.2,
                                                    share_weights=False)
        #self.node_update = MLP([self.representation_size]*2, self.representation_size*5,dropout=dropout)
        self.node_update = MLP([self.representation_size]*2, self.representation_size*3,dropout=dropout)
        
        # Edge related layers
        self.edge_encoder = MLP([self.representation_size]*2, n_edge_features)
        self.edge_update = EdgeModelLtp(self.representation_size*2, #Node features
                                        self.representation_size*2, #Edge features
                                        0, #No global features 
                                        self.representation_size, #hidden features
                                        dropout=dropout)

        # Global related layers
        self.global_encoder  = MLP([self.representation_size]*2,n_global_features)
        self.global_update = GlobalModel(self.representation_size*3,self.representation_size,dropout=dropout)
        
    def forward(self, batch):
        node_data  = batch['node'].x
        node_index = batch['node'].batch
        #edge_index = torch.repeat_interleave(torch.arange(0, len(batch['globals'].x)).cuda(), batch['n_edge'].x)
        edge_index = torch.repeat_interleave(torch.arange(0, len(batch['globals'].x),device=self.device), batch['n_edge'].x)

        edge_features_node = batch['node','sends','node'].edge_attr
        edge_features_node_index = batch['node','sends','node'].edge_index
        global_data = batch['globals'].x
        
        node_data = self.node_encoder(node_data)
        edge_features_node = self.edge_encoder(edge_features_node)
        global_data = self.global_encoder(global_data)

        node_data_original = node_data
        edge_features_node_original = edge_features_node
        #global_data_original = global_data

        for i in range(self.num_rounds) :
            node_data = torch.cat([node_data_original,node_data],dim=1)
            edge_features_node = torch.cat([edge_features_node_original,edge_features_node],dim=1)
            #global_data = torch.cat([global_data_original,global_data],dim=1)

            src = node_data[edge_features_node_index[0]]
            dest = node_data[edge_features_node_index[1]]
            #global_node_repeat = global_data[node_index]
            #global_edge_repeat = global_data[edge_index]

            edge_features_node = self.edge_update(src,dest,edge_features_node,None)
            node_data = self.node_update(self.node_attention_layer(node_data,edge_features_node,edge_features_node_index[1],None))
            #global_data = self.global_update(node_data,edge_index,edge_features_node,global_data,node_index)

        global_data = self.global_update(node_data,edge_index,edge_features_node,global_data,node_index)

        return node_data, edge_features_node,global_data

class EncodeDecode(nn.Module):
    def __init__(self, n_features, n_edge_features,n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 attn_dropout ,
                 action_space,
                 batch_size,
                 n_heads,
                 g_node,
                 num_mlp_layers_gnn,
                 device,
                 action_options,
                 object_options):
        super(EncodeDecode,self).__init__()

    def __str__(self):
        return f"{self.__class__.__name__}"

    def get_best_action_scores_locations_old(self,a_scores,k):
        all_actions_batches = []
        all_actions_scores = []
        for i in range(a_scores.shape[0]):
            values , indexes = torch.topk(a_scores[i],k)
            all_actions_batches.append(indexes)
            all_actions_scores.append(values)

        return all_actions_batches,all_actions_scores

    def get_best_action_scores_locations(self, a_scores, k):
        # Get top-k action scores and their indices using tensor operations.
        # Apply topk across all batches at once
        # Returns (values, indices) both of shape [batch_size, min(k, num_actions)]
        # k can exceed the schema count when a multi-domain model decodes a
        # schema-poor domain (e.g. Visitall has 1 schema): clamp to width.
        k = min(k, a_scores.shape[1])
        values, indices = torch.topk(a_scores, k, dim=1)
        return indices, values

    def get_best_action_object_scores_locations_old(self,ao_scores,n_node,k):
        all_objects_batches = []
        all_objects_scores = []

        for i in range(ao_scores.shape[0]) :
            #Current graph objects is just a safety net to ensure we don't go over the number of objects in the graph
            current_graph_objects = n_node[int(i/self.max_number_action_parameters)]
            values , indexes = torch.topk(ao_scores[i][:current_graph_objects],k)
            all_objects_batches.append(indexes)
            all_objects_scores.append(values)

        return all_objects_batches, all_objects_scores

    def get_best_action_object_scores_locations(self, ao_scores, n_node, k, n_objects=None,
                                                n_parameters=None):
        """
        Get top-k action-object scores and their indices using tensor operations.
        Args:
            ao_scores: Tensor of [sum(n_parameters), max_nodes] - one row per
                (graph, parameter slot). Rows per graph are NOT uniform in
                general; pass n_parameters so the row->graph map is exact.
            n_node: Tensor containing total number of nodes per graph (including all node types)
            k: Number of top objects to select
            n_objects: Per-graph object counts; when given, selection is
                restricted to object nodes (see comment below). None keeps
                the legacy all-nodes mask (training targets path).
            n_parameters: Per-graph parameter-slot counts. When given, the
                row->graph map is built from them; when omitted the legacy
                uniform-stride map is used (see below).

        Returns:
            Tuple of (indices_tensor, values_tensor)
        """
        batch_size = ao_scores.shape[0]

        # Create a mask to ignore scores beyond valid nodes for each graph
        max_nodes = ao_scores.shape[1]

        # Which graph does each row of ao_scores belong to?
        #
        # The legacy answer, row // max_number_action_parameters, assumes every
        # graph contributes exactly max_number_action_parameters rows. That is
        # false whenever the batch's rows-per-graph differs from the decoder's
        # arity cap, and it is how a 200-graph batch came to be read as 67
        # graphs (RuntimeError in get_best_object_embeddings_ltp: "size of
        # tensor a (200) must match tensor b (67)"). compute_object_scores has
        # always derived this from n_parameters; do the same here when the
        # caller supplies them, and keep the old map otherwise so the training
        # path and ablations.py are untouched.
        if n_parameters is not None:
            n_par = n_parameters.to(torch.long)
            graph_indices = torch.repeat_interleave(
                torch.arange(n_par.numel(), device=ao_scores.device), n_par)
            if graph_indices.numel() != batch_size:
                raise RuntimeError(
                    f"ao_scores has {batch_size} rows but n_parameters sums to "
                    f"{int(n_par.sum())} over {n_par.numel()} graphs; the batch "
                    f"and its parameter counts disagree")
        else:
            graph_indices = torch.div(torch.arange(batch_size, device=ao_scores.device),
                                    self.max_number_action_parameters, rounding_mode='floor').long()

        # Get number of nodes for each row in ao_scores
        # n_node represents total nodes in the graph (including object, predicate, action nodes)
        row_n_nodes = n_node[graph_indices]

        # Create range tensor for masking
        node_indices = torch.arange(max_nodes, device=ao_scores.device).expand(batch_size, -1)

        # Action parameters are always OBJECTS (the first n_objects nodes of
        # each graph). compute_object_scores writes real scores (which can be
        # NEGATIVE) only into object columns and leaves literal/schema columns
        # at exactly 0, so masking by n_node lets topk select a 0-scored
        # literal node whenever every object scores below zero - producing a
        # malformed action. Never triggered by trained in-domain scores
        # (correct objects score positive) but systematic in zero-shot eval.
        # Pass n_objects to restrict selection to real objects.
        if n_objects is not None:
            valid_counts = n_objects[graph_indices]
        else:
            valid_counts = row_n_nodes
        mask = node_indices < valid_counts.unsqueeze(1)
        
        # Apply mask to scores (set invalid scores to -inf)
        masked_scores = torch.where(mask, ao_scores, torch.tensor(float('-inf'), device=ao_scores.device))
        
        # Get top-k values and indices
        values, indices = torch.topk(masked_scores, k, dim=1)
        
        return indices, values

    def get_best_action_embeddings_old(self,x,all_actions,n_node,domain_number_actions):
        #required_correct_features = torch.zeros((len(all_actions),1,self.representation_size),dtype=torch.float32).cuda()
        required_correct_features = torch.zeros((len(all_actions),1,self.representation_size),dtype=torch.float32,device=self.device)
        current_number_nodes = 0
        for a,action in enumerate(all_actions) :
            action_curr_graph = n_node[a] - domain_number_actions
            required_correct_features[a][0] = x[current_number_nodes+action_curr_graph+action][:]
            current_number_nodes += n_node[a]
        return required_correct_features#,number_action_parameters

    def _decode_action_count(self, n_actions):
        """Schema-node count used to locate the selected schema's embedding.

        Each graph lays out its schema nodes as the LAST n_action nodes, so
        the per-graph count from the data (n_actions) is the correct offset.
        Models trained before this fix used len(model action space) instead,
        which points below the schema block whenever a graph's domain has
        fewer schemas than the (merged multi-domain) training action space -
        the GRU was then conditioned on arbitrary literal-node embeddings.
        Set GABAR_LEGACY_ACTION_OFFSET=1 when testing checkpoints trained
        with that behavior; train/test offsets must match.
        """
        if os.environ.get('GABAR_LEGACY_ACTION_OFFSET', '') == '1':
            return self.number_actions
        return n_actions

    def get_best_action_embeddings(self,x,all_actions,n_node,domain_number_actions):
        # 1. Calculate the starting index of each graph in the flattened tensor `x`
        offsets = torch.zeros_like(n_node) # Shape [num_graphs]
        # Calculate cumulative sum of nodes *before* the current graph
        offsets[1:] = torch.cumsum(n_node[:-1], dim=0)

        action_block_starts_relative = n_node - domain_number_actions

        # 3. Calculate the absolute index in `x` for the selected action in each graph
        absolute_indices = offsets + action_block_starts_relative + all_actions

        # 4. Gather the features using efficient indexing
        selected_features = torch.index_select(x, dim=0, index=absolute_indices)

        # 5. Reshape to match the original output format [num_graphs, 1, representation_size]
        output_features = selected_features.unsqueeze(1)

        return output_features

    def compute_action_scores_old(self,x,n_actions,hidden_state, action_idxs):
        '''
        Computing the score for each of the actions (Updating the graph[action_scores]
        for all actions
            - each super graph has a certain number of action nodes (Number of graphs * num of actions per graph)
            - We score each set of actions in a graph (every diff value in global index refers to a diff graph)
            - Hence we compute scores for a set of each graph - over all actions and store it in action_scores (1 vector per graph)
        '''
        #number_actions_array = [torch.tensor(0).cuda()]
        number_actions_array = [torch.tensor(0,device=self.device)]

        for elem in n_actions :
            number_actions_array.append(number_actions_array[-1]+elem)

        return torch.stack([torch.matmul(x[action_idxs[int(number_actions_array[i]):int(number_actions_array[i + 1])]],hidden_state[-1,i]) for i in range(len(number_actions_array)-1)])

    def compute_action_scores(self, x, n_actions, hidden_state, action_idxs):
        """
        Completely vectorized implementation to score all actions for each graph.

        Args:
            x (Tensor): Features tensor of shape [num_total_nodes, feature_dim]
            n_actions (Tensor): Number of actions per graph, tensor of shape [num_graphs]
            hidden_state (Tensor): Hidden state tensor of shape [seq_len, num_graphs, hidden_dim]
            action_idxs (Tensor): Action indices tensor of shape [num_total_actions]

        Returns:
            Tensor: Tensor of action scores, shape [num_graphs, max_actions], padded with -inf.
        """
        # Ensure inputs are tensors on the correct device
        device = x.device # Use device from input tensor x

        num_graphs = hidden_state.size(1) # Get num_graphs from hidden_state dim 1
        total_actions = action_idxs.size(0)

        # --- Calculation Steps ---

        # 1. Get relevant hidden state (last timestep)
        relevant_hidden = hidden_state[-1]  # Shape: [num_graphs, hidden_dim]

        # 2. Get all action features
        action_features = x[action_idxs]  # Shape: [total_actions, feature_dim]

        # 3. Create graph indices directly using repeat_interleave
        #    This correctly maps each action in the flat list to its graph index.
        graph_indices = torch.arange(num_graphs, device=device).repeat_interleave(n_actions)

        # 4. Get the corresponding hidden state for each action's graph
        #    Shape: [total_actions, hidden_dim]
        hidden_for_actions = relevant_hidden[graph_indices]

        # 5. Compute dot product between action features and their corresponding hidden states
        #    Shape: [total_actions]
        action_scores_flat = torch.sum(action_features * hidden_for_actions, dim=1)

        # 6. Calculate local indices (0 to n_actions[g]-1 for each graph g)
        #    Create cumulative sum of actions *excluding* the current graph's count
        #    to find the starting index of each graph's actions.
        cum_actions_start = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            torch.cumsum(n_actions[:-1], dim=0, dtype=torch.long) # Exclude last element for start indices
        ])
        # The starting index for each action's graph block
        graph_start_indices = cum_actions_start[graph_indices]
        # Create a global range for actions
        action_range = torch.arange(total_actions, device=device)
        # Subtract the start index to get the 0-based index within the graph
        local_indices = action_range - graph_start_indices

        # 7. Prepare output tensor
        #    Find max number of actions for padding dimension. Handle case where n_actions is empty.
        max_actions = torch.max(n_actions).item() if num_graphs > 0 and n_actions.numel() > 0 else 0
        # Initialize with -inf for proper padding / softmax compatibility
        action_scores = torch.full((num_graphs, max_actions), float('-inf'), device=device)

        # 8. Use scatter or index_put_ to place scores in the correct positions
        #    Create the 2D indices for index_put_
        #    row indices are graph_indices, column indices are local_indices
        indices_tuple = (graph_indices, local_indices)

        # Place the computed scores into the output tensor
        action_scores.index_put_(indices_tuple, action_scores_flat)

        return action_scores

    def extract_graph_info_ltp(self,data):
        #torch_time = time.time()
        action_idxs = torch.where(data['node'].x[:,0] == 1)[0]
        object_idxs = torch.where(data['node'].x[:,1] == 1)[0]

        a_scores = data['action_scores'].x
        ao_scores = data['action_object_scores'].x
        n_node = data['n_node'].x
        n_parameters = data['n_parameters'].x
        n_actions = data['n_action'].x
        n_objects = data['n_object'].x
        number_graphs = data['n_node'].x.shape[0]
        #torch_where_time = time.time() - torch_time
        return action_idxs, object_idxs, a_scores, ao_scores, n_node, n_parameters, n_actions, n_objects,number_graphs

    def compute_object_scores(self, x, n_params, n_objects, ao_scores, hidden_state, object_idxs, parameter_number):
        # Get dimensions
        total_params = ao_scores.shape[0]  # Total number of parameters across all graphs
        max_length = ao_scores.shape[1]    # Maximum length dimension
        
        # Step 1: Compute parameter indices correctly
        # In the original code, current_parameter_indexes represents the indices
        # for the specific parameter_number across all graphs
        cumulative_params = torch.cumsum(n_params, dim=0)
        param_start_indices = cumulative_params - n_params
        current_parameter_indexes = (param_start_indices + parameter_number).to(torch.long)
        
        # Step 2: Create a mask for the parameters we're interested in
        mask_matrix = torch.zeros(total_params, max_length, device=self.device)
        mask_matrix[current_parameter_indexes, :] = 1
        
        # Step 3: Create object ranges for each graph
        cumulative_objects = torch.cat([torch.tensor([0], device=self.device), 
                                    torch.cumsum(n_objects, dim=0)])
        
        # Get start and end indices for objects in each graph
        object_start_idxs = cumulative_objects[:-1]
        object_end_idxs = cumulative_objects[1:]
        
        # Repeat these ranges according to n_params for each graph
        object_start_idxs_repeated = torch.repeat_interleave(object_start_idxs, n_params)
        object_end_idxs_repeated = torch.repeat_interleave(object_end_idxs, n_params)
        
        # Step 4: Repeat hidden state for the matrix multiplication
        new_hidden = torch.repeat_interleave(hidden_state[-1], n_params, dim=0)
        
        # Step 5: Determine the maximum number of objects across all graphs
        object_counts = object_end_idxs_repeated - object_start_idxs_repeated
        max_obj_count = torch.max(object_counts).item()
        
        # Step 6: Create object indices and masks for all batch items at once
        # Create a range tensor that will be used for indexing
        indices_range = torch.arange(max_obj_count, device=self.device).unsqueeze(0).repeat(total_params, 1)
        
        # Create masks for valid object indices
        object_masks = indices_range < object_counts.unsqueeze(1)
        
        # Calculate actual indices into object_idxs
        # First, create base indices by adding start indices to the range
        object_indices = object_start_idxs_repeated.unsqueeze(1) + indices_range
        
        # Zero out invalid indices
        object_indices = torch.where(object_masks, object_indices, torch.zeros_like(object_indices))
        
        # Step 7: Ensure all indices are within bounds
        obj_idx_length = object_idxs.size(0)
        safe_indices = torch.clamp(object_indices, 0, obj_idx_length - 1)
        
        # Get the actual object indices from object_idxs
        actual_obj_idxs = object_idxs[safe_indices]
        
        # Apply mask to zero out invalid indices
        masked_obj_idxs = torch.where(object_masks, actual_obj_idxs, torch.zeros_like(actual_obj_idxs))
        
        # Step 8: Gather features from x using the masked object indices
        object_features = x[masked_obj_idxs]
        
        # Apply mask to zero out invalid features
        object_features = object_features * object_masks.unsqueeze(-1)
        
        # Step 9: Compute scores using batch matrix multiplication
        hidden_expanded = new_hidden.unsqueeze(1)  # [total_params, 1, dim]
        scores = torch.bmm(object_features, hidden_expanded.transpose(1, 2)).squeeze(-1)  # [total_params, max_obj]
        
        # Apply mask to scores
        scores = scores * object_masks

        variable_action_object_scores = torch.zeros(total_params, max_length, device=self.device)
        copy_length = min(max_obj_count, max_length)
        
        if copy_length > 0:
            variable_action_object_scores[:, :copy_length] = scores[:, :copy_length]
        
        # Step 10: Apply parameter mask to final scores
        return torch.mul(mask_matrix, variable_action_object_scores)

    def get_best_object_embeddings_old(self,x,all_objects,all_actions,parameter_number,n_params,n_node):
        current_number_nodes = 0
        objects_counter = parameter_number
        feature_captured_object_counter = 0
        required_correct_object_features = torch.zeros((len(all_actions), 1, self.representation_size),
                                                       dtype=torch.float32,device=self.device)
        for a, action in enumerate(all_actions):
            object_idx = all_objects[objects_counter]
            required_correct_object_features[feature_captured_object_counter][0] = x[
                                                                      current_number_nodes + object_idx][:]
            objects_counter += int(n_params[a])
            feature_captured_object_counter += 1
            current_number_nodes += n_node[a]
        return required_correct_object_features

    def get_best_object_embeddings(self,x,all_objects,all_actions,parameter_number,n_params,n_node):
        # Explanation 
        # Fisrt, we extract the starting location for obj nodes in each graph
        # Then, we get all the object positions (we get objects for all parameters, we extract current parameter objs)
        return x[
            torch.cat((n_node.new_zeros(1), torch.cumsum(n_node[:-1], 0)))
            + all_objects[torch.cumsum(n_params, 0) - n_params[0] + parameter_number]
        ].unsqueeze(1)

    def get_best_object_embeddings_ltp_old(self,x,all_objects,n_node, num_graphs):
        current_number_nodes = 0
        #required_correct_object_features = torch.zeros((num_graphs, 1, self.representation_size),
        #                                               dtype=torch.float32).cuda()
        required_correct_object_features = torch.zeros((num_graphs, 1, self.representation_size),
                                                       dtype=torch.float32,device=self.device)

        for i in range(num_graphs):
            object_idx = all_objects[i]
            required_correct_object_features[i][0] = x[current_number_nodes+object_idx][:]
            current_number_nodes += n_node[i]
        return required_correct_object_features

    def get_best_object_embeddings_ltp(self,x,all_objects,n_node, num_graphs):
        return x[
            torch.cat((n_node.new_zeros(1), torch.cumsum(n_node[:-1], 0)))
            + all_objects
        ].unsqueeze(1)
    
    #U is the global value
    def non_beam_decode_non_CD(self,x, u ,a_scores_new,ao_scores,n_node,n_parameters,n_objects,object_idxs,n_actions,action_idxs, num_graphs):
        ao_scores_new = torch.zeros(ao_scores.shape,device=self.device)
        for parameter_num in range (self.max_number_action_parameters):
            obj_intermediate = self.object_score_decoder(
                                        torch.cat([u, 
                                        F.one_hot(torch.full((num_graphs,), parameter_num, device=self.device), 
                                                  num_classes=self.max_number_action_parameters)
                                                  ], dim=1))

            obj_intermediate = obj_intermediate.unsqueeze(0)
            ao_scores_new += self.compute_object_scores(x, n_parameters,n_objects, ao_scores,
                                                        obj_intermediate,
                                                       object_idxs,parameter_num)
        
        return a_scores_new, ao_scores_new

    def beam_search_non_CD(self, x, u , number_graphs,ao_scores, a_scores_new,n_node,
                    n_parameters,n_objects,object_idxs,n_actions,
                    action_idxs):
        a_scores_final, ao_scores_final = self.non_beam_decode_non_CD(x, u ,a_scores_new,ao_scores,n_node,
                                    n_parameters,n_objects,object_idxs,n_actions,action_idxs,number_graphs)

        self.max_num_actions = self.action_options
        self.max_num_objects = self.object_options
        results = []
        
        # Get top-k actions
        top_action_values, top_action_indices = torch.topk(a_scores_final, min(self.max_num_actions,a_scores_final.shape[1] ))
        
        # Get top-k objects for each parameter position
        max_objects_per_action = ao_scores_final.shape[0]
        top_object_values, top_object_indices = torch.topk(ao_scores_final, min(self.max_num_objects, ao_scores_final.shape[1]), dim=1)
        
        # Process each top action
        for i, action_idx in enumerate(top_action_indices[0]):
            action_idx = action_idx.item()
            action_score = top_action_values[0][i]
            
            # Get number of parameters for this action
            num_params = self.action_parameter_number_dict.get(action_idx, 0)
            num_params = min(num_params, max_objects_per_action)
            
            if num_params == 0:
                # If action requires no objects, simply add it
                results.append((action_score, [torch.tensor(action_idx, device=self.device)]))
                continue
            
            # Get all combinations of objects for this action
            # Only consider parameters that this action needs
            param_object_indices = [top_object_indices[j][:self.max_num_objects] for j in range(num_params)]
            
            # Generate all possible combinations of objects
            object_combinations = list(itertools.product(*param_object_indices))
            
            # Create tuples for all combinations with this action
            for obj_combo in object_combinations:
                selected_indices = [torch.tensor(action_idx, device=self.device)]
                selected_indices.extend([obj.to(self.device) for obj in obj_combo])
                results.append((action_score, selected_indices))
        
        #SCORE IS NEVER USED HENCE being ignored here
        return results

        

#class GNN_GRU(nn.Module):
class GNN_GRU(EncodeDecode):
    def __init__(self, n_features, n_edge_features,n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 attn_dropout ,
                 action_space,
                 batch_size,
                 n_heads,
                 g_node,
                 num_mlp_layers_gnn,
                 device,
                 action_options,
                 object_options,
                 ablation,
                 ):
        super(GNN_GRU,self).__init__(
            n_features, n_edge_features, n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 attn_dropout ,
                 action_space,
                 batch_size,
                 n_heads,
                 g_node,
                 num_mlp_layers_gnn,
                 device,
                 action_options,
                 object_options,
        )
        self.max_num_actions = 1
        self.max_num_objects = 1
        self.device = device

        if g_node is True :
            self.encoder = HeteroGNN_global(n_features,n_edge_features,n_global_features\
                                        ,n_hidden,dropout,attn_dropout,gnn_rounds,n_heads,num_mlp_layers_gnn,device)
            #self.encoder = HeteroGNN_global_Wrapper(n_features,n_edge_features,n_global_features\
            #                            ,n_hidden,dropout,attn_dropout,gnn_rounds,n_heads,device)
        else :
            self.encoder = HeteroGNN(n_features,n_edge_features,n_global_features\
                                        ,n_hidden,dropout,attn_dropout,gnn_rounds,n_heads,device)
        
        #self.encoder = torch.jit.script(self.encoder)
        self.representation_size = n_hidden
        self.max_number_action_parameters = 0
        self.action_parameter_number_dict = {}
        self.number_actions = len(action_space.keys())
        self.action_options = min(action_options, self.number_actions)
        self.object_options = object_options
        number_graphs = batch_size

        if action_space != None:
            i = 0
            for key, values in action_space.items():
                self.max_number_action_parameters = max(len(values.params),\
                                                        self.max_number_action_parameters)
                self.action_parameter_number_dict[i] = len(values.params)
                i += 1
            self.number_actions = len(action_space.keys())

        self.decoder = nn.GRU(n_hidden, hidden_size=n_hidden,\
                              num_layers=num_decoder_layers,bias=False,batch_first=True)

        self.num_decoder_layers = num_decoder_layers
        self.action_score_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            #nn.ReLU(),
            #nn.Linear(n_hidden, n_hidden),
            #nn.LayerNorm(n_hidden),
        )
        self.training_mode = True
        self.ablation = ablation
        if self.ablation == "main_val":
            self.global_val = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(n_hidden, int(n_hidden/2)),
                nn.BatchNorm1d(int(n_hidden/2)),
                nn.ReLU(),
                nn.Linear(int(n_hidden/2), int(n_hidden/4)),
                nn.BatchNorm1d(int(n_hidden/4)),
                nn.ReLU(),
                nn.Linear(int(n_hidden/4), int(n_hidden/8)),
                nn.BatchNorm1d(int(n_hidden/8)),
                nn.ReLU(),
                nn.Linear(int(n_hidden/8), 1),
            )

    def __str__(self):
        return f"{self.__class__.__name__}"

    #def forward(self,x,edge_idx,edge_attr,u,a_scores, ao_scores, batch=None):
    def extract_data_and_run_encoder(self,data):
        graph_info = self.extract_graph_info_ltp(data)
        action_idxs, object_idxs, a_scores, ao_scores, n_node, n_parameters, n_actions, n_objects,number_graphs = graph_info 
        h0 = torch.zeros(self.num_decoder_layers,number_graphs,self.representation_size,device=self.device)
        x,edge_attr, u = self.encoder(data)
        u = u.unsqueeze(1)
        _,hidden_state = self.decoder(u,h0)
        x = self.action_score_decoder(x)
        return x, u,  hidden_state, a_scores, ao_scores, n_node, n_parameters, n_objects,object_idxs,n_actions,action_idxs,number_graphs

    def forward(self,data, beam_search = False):
        x, u, hidden_state, a_scores, ao_scores, n_node, n_parameters, n_objects,object_idxs,n_actions,action_idxs,number_graphs = self.extract_data_and_run_encoder(data)
        return self.non_beam_decode(x,hidden_state,a_scores,ao_scores,n_node,
                                    n_parameters,n_objects,object_idxs,n_actions,action_idxs)
        #return self.beam_search_parallel(x,hidden_state,number_graphs=number_graphs,
        #                ao_scores=ao_scores,n_node=n_node,n_parameters=n_parameters,
        #                n_objects=n_objects,object_idxs=object_idxs,n_actions=n_actions,
        #                action_idxs=action_idxs)
        #return self.non_beam_decode(x,hidden_state,a_scores,ao_scores,n_node,
        #                            n_parameters,n_objects,object_idxs,n_actions,action_idxs)
    def forward_with_parallel_beam_search(self,data, beam_search = False):
        x, u, hidden_state, a_scores, ao_scores, n_node, n_parameters, n_objects,object_idxs,n_actions,action_idxs,number_graphs = self.extract_data_and_run_encoder(data)
        return self.beam_search_parallel(x,hidden_state,number_graphs=number_graphs,
                        ao_scores=ao_scores,n_node=n_node,n_parameters=n_parameters,
                        n_objects=n_objects,object_idxs=object_idxs,n_actions=n_actions,
                        action_idxs=action_idxs)

    def forward_beam_decode(self,data):
        # CAN ONLY BE USED WITH BATCH SIZE 1
        x, u, hidden_state, a_scores, ao_scores, n_node, n_parameters, n_objects,object_idxs,n_actions,action_idxs,number_graphs = self.extract_data_and_run_encoder(data)
        return self.beam_search_v2(x,hidden_state,number_graphs=number_graphs,
                        ao_scores=ao_scores,n_node=n_node,n_parameters=n_parameters,
                        n_objects=n_objects,object_idxs=object_idxs,n_actions=n_actions,
                        action_idxs=action_idxs)

    def forward_with_value_before_decoding(self,data):
        x, u, hidden_state, a_scores, ao_scores, n_node, n_parameters, n_objects,object_idxs,n_actions,action_idxs,number_graphs = self.extract_data_and_run_encoder(data)
        #state_val = self.global_val(u.squeeze(1))
        state_val = self.global_val(u.view(u.size(0), -1))
        return x,u,  hidden_state, a_scores, ao_scores, n_node, n_parameters, n_objects,object_idxs,n_actions,action_idxs,number_graphs, state_val 

    def forward_with_value(self,data):
        x,u, hidden_state, a_scores, ao_scores, n_node, n_parameters, n_objects, object_idxs, n_actions, action_idxs,number_graphs, state_val = self.forward_with_value_before_decoding(data)
        return self.non_beam_decode(x,hidden_state,a_scores,ao_scores,n_node,
                                    n_parameters,n_objects,object_idxs,n_actions,action_idxs), state_val

    def forward_with_value_beam_search(self,data):
        # CAN ONLY BE USED WITH BATCH SIZE 1
        x,u, hidden_state, a_scores, ao_scores, n_node, n_parameters, n_objects, object_idxs, n_actions, action_idxs,number_graphs, state_val = self.forward_with_value_before_decoding(data)
        return self.beam_search_v2(x,hidden_state,number_graphs=number_graphs,
                        ao_scores=ao_scores,n_node=n_node,n_parameters=n_parameters,
                        n_objects=n_objects,object_idxs=object_idxs,n_actions=n_actions,
                        action_idxs=action_idxs), state_val

    def forward_with_value_parallel_beam_search(self,data):
        x,u ,hidden_state, a_scores, ao_scores, n_node, n_parameters, n_objects, object_idxs, n_actions, action_idxs,number_graphs, state_val = self.forward_with_value_before_decoding(data)
        return self.beam_search_parallel(x,hidden_state,number_graphs=number_graphs,
                        ao_scores=ao_scores,n_node=n_node,n_parameters=n_parameters,
                        n_objects=n_objects,object_idxs=object_idxs,n_actions=n_actions,
                        action_idxs=action_idxs),state_val


    def non_beam_decode(self,x,hidden_state,a_scores,ao_scores,n_node,n_parameters,n_objects,object_idxs,n_actions,action_idxs):
        all_actions_batches,_ = self.get_best_action_scores_locations(a_scores,self.max_num_actions)
        all_objects_batches,_ = self.get_best_action_object_scores_locations(ao_scores,n_node,self.max_num_objects)
        
        all_actions = all_actions_batches[:, 0]
        all_objects = all_objects_batches[:, 0]

        #action_scores_time = time.time()
        a_scores_new = self.compute_action_scores(x,n_actions,hidden_state,action_idxs)
        #action_scores_total_time = time.time() - action_scores_time

        #computing_best_action_embedding = time.time()
        decoder_input = self.get_best_action_embeddings(x,all_actions,n_node,domain_number_actions=self._decode_action_count(n_actions))
        #computing_best_action_embedding_time = time.time() - computing_best_action_embedding

        #decoder_time = time.time()
        #ao_scores_new = torch.zeros(ao_scores.shape).cuda()
        ao_scores_new = torch.zeros(ao_scores.shape,device=self.device)
        action_object_scores_0 = []

        for i in range(0, self.max_number_action_parameters):
            _, hidden_state = self.decoder(decoder_input, hidden_state) 
            ao_scores_new += self.compute_object_scores(x, n_parameters,n_objects, ao_scores,hidden_state,
                                                       object_idxs,i)
            #computing_best_object_embedding = time.time()
            if i == self.max_number_action_parameters-1 :
                break
            decoder_input = self.get_best_object_embeddings(x, all_objects, all_actions,parameter_number=i,
                                                            n_params=n_parameters,
                                                            n_node=n_node)   

            #computing_best_object_embedding_time = time.time() - computing_best_object_embedding
        #decoder_total_time_2 = time.time()-decoder_time
        #return a_scores, ao_scores
        return a_scores_new, ao_scores_new

        #end_time = time.time()
        #print ("encoder time : ", encoder_time)
        #print ("function time : ", end_time-start_time)
        '''
        print ("fraction of time in encoding", encoder_time/(end_time-start_time))
        print ("fraction of time in decoding",decoder_total_time/(end_time-start_time))
        print ("fraction of time in decodi 2", decoder_total_time_2/(end_time-start_time))
        print ("Fraction of time torch where", torch_where_time/(end_time-start_time))
        print ("Fraction of time action embe", computing_best_action_embedding_time/(end_time-start_time))
        print ("Fraction of time obj embed  ", computing_best_object_embedding_time/(end_time-start_time))
        print ("actual time spent decoding", decoder_total_time)
        print ("actual time spent decoding", decoder_total_time_2)
        print ("Total TIME : ", end_time-start_time)
        '''
    
    @torch.no_grad()  
    def beam_search_v2(self, x, hidden_state, number_graphs,ao_scores,n_node,
                    n_parameters,n_objects,object_idxs,n_actions,
                    action_idxs):

        a_scores_new = self.compute_action_scores(x,n_actions,hidden_state,action_idxs)
        # Clamp to the graph's schema count: a multi-domain model's
        # action_options can exceed it on schema-poor domains (Visitall: 1),
        # and the loop below indexes columns up to max_num_actions.
        self.max_num_actions = min(self.action_options, a_scores_new.shape[1])
        self.max_num_objects = self.object_options
        all_actions_batches,all_actions_scores = self.get_best_action_scores_locations(a_scores_new,self.max_num_actions)

        # Active beams now contain (token_ids, embeddings, scores)
        active_beams = []
        
        # Storage for completed sequences
        finished_beams = []
        finished_scores = []
        curr_depth = 0

        for action_idx in range(self.max_num_actions):
            decoder_input = self.get_best_action_embeddings(x,all_actions_batches[:,action_idx],n_node,domain_number_actions=self._decode_action_count(n_actions))
            all_curr_action_scores = [elem[action_idx] for elem in all_actions_scores]
            active_beams.append(([all_actions_batches[:,action_idx][0]], decoder_input, 
                                 all_curr_action_scores[0], hidden_state, curr_depth ))

        for parameter_number in range(self.max_number_action_parameters):  # replace with the actual maximum sequence length
            if not active_beams:
                break

            # Prepare batched inputs for the model
            current_decoder_input = torch.cat([beam[1] for beam in active_beams], dim=0)
            current_scores = [beam[2] for beam in active_beams]
            current_hidden = torch.cat([beam[3] for beam in active_beams], dim=1)
            curr_depth = active_beams[0][4]
            _, new_hidden = self.decoder(current_decoder_input, current_hidden)

            #new_hidden_split = torch.split(new_hidden, split_size_or_sections=1, dim=1)

            # Prepare for next step
            new_active_beams = []
            # compute_object_scores reads only .shape of this argument and
            # returns a fresh tensor, so allocating and zero-filling one here
            # was pure waste, once per parameter slot per decode. ao_scores has
            # the same shape; nothing reads its contents either.
            ao_scores_new = ao_scores

            # Grounding-constrained decoding (GABAR_CONSTRAINED_DECODE=1):
            # _decode_allowed maps (schema, objects chosen so far) -> the object
            # nodes that some APPLICABLE grounding puts in the next slot. The
            # applicable set is already enumerated for the graph, so this costs
            # a dict lookup and makes every completed action applicable by
            # construction (V1 = 100%). Without it the decoder can compose
            # (schema, objects) freely and spend its beam on actions that do
            # not exist -- e.g. drive-truck(package, package, loc, city).
            _allowed = getattr(self, "_decode_allowed", None)

            for beam_idx, beam in enumerate(active_beams):
                ao_scores_new = self.compute_object_scores(x, n_parameters,n_objects,
                                                            ao_scores_new,
                                                            #new_hidden_split[beam_idx],
                                                            new_hidden[:, beam_idx:beam_idx+1],
                                                        object_idxs,parameter_number)

                ok_nodes = None
                if _allowed is not None:
                    seq = active_beams[beam_idx][0]
                    key = (int(seq[0]), tuple(int(t) for t in seq[1:]))
                    ok_nodes = _allowed.get(key) or set()
                    # Score -inf outside the allowed set so top-k is drawn from
                    # legal objects; rows are [param, node] and single-graph
                    # decode puts this parameter on row `parameter_number`.
                    if parameter_number < ao_scores_new.shape[0]:
                        row = ao_scores_new[parameter_number]
                        keep = torch.zeros_like(row, dtype=torch.bool)
                        if ok_nodes:
                            keep[torch.tensor(sorted(ok_nodes), dtype=torch.long,
                                              device=row.device)] = True
                        row[~keep] = float("-inf")

                all_objects_batches_all_params,all_objects_scores_all_params = self.get_best_action_object_scores_locations(
                                        ao_scores=ao_scores_new, n_node=n_node, k=self.max_num_objects,
                                        n_objects=n_objects)

                all_objects_scores = all_objects_scores_all_params[parameter_number]
                all_objects_batches = all_objects_batches_all_params[parameter_number]
                for object_option in range(self.max_num_objects):
                    all_objects = all_objects_batches[object_option]
                    all_curr_obj_scores = all_objects_scores[object_option]
                    # Fewer legal objects than k: top-k still returns k rows,
                    # the surplus being masked-out ones. Drop them.
                    if ok_nodes is not None and int(all_objects) not in ok_nodes:
                        continue
                    new_decoder_input = self.get_best_object_embeddings_ltp(x, 
                                                                            all_objects,
                                                                             n_node, number_graphs)
                    

                    curr_sequence = active_beams[beam_idx][0] + [all_objects] 
                    action = curr_sequence[0].item()
                    new_score = (current_scores[beam_idx]*(curr_depth+1) + all_curr_obj_scores)/ (curr_depth + 2 )
                    if curr_depth + 1 == self.action_parameter_number_dict[action]:
                        finished_beams.append(curr_sequence)
                        finished_scores.append(new_score)
                        continue

                    new_active_beams.append((curr_sequence, new_decoder_input, 
                                             new_score, 
                                             #new_hidden_split[beam_idx],
                                             new_hidden[:, beam_idx:beam_idx+1],
                                             curr_depth+1))
            
            active_beams = new_active_beams[:]

        # Sort using tensor operations instead of python lists 
        scores_tensor = torch.tensor(finished_scores, device=self.device)
        indices = torch.argsort(scores_tensor, descending=True)
        sorted_beams = [finished_beams[i] for i in indices.tolist()]
        sorted_scores = scores_tensor[indices].tolist()
        results = list(zip(sorted_scores, sorted_beams))
        return results

    @torch.no_grad()  
    def beam_search_parallel(self, x, hidden_state, number_graphs,ao_scores,n_node,
                    n_parameters,n_objects,object_idxs,n_actions,
                    action_idxs):

        # n_parameters is authoritative for how ao_scores' rows map to graphs;
        # everything below indexes through it rather than through the decoder's
        # arity cap. Hoisted here because it is invariant over both loops, and
        # in a launch-bound decoder the cost of these is the launch, not the
        # arithmetic. No .item()/.sum() readback: that would be a device sync
        # per decode. get_best_action_object_scores_locations raises on a
        # row-count disagreement, and its check is free there.
        n_par_long = n_parameters.to(torch.long)
        _par_starts = torch.cumsum(n_par_long, 0) - n_par_long   # first row of each graph
        _par_last = (n_par_long - 1).clamp(min=0)                # its last, for clamping

        a_scores_new = self.compute_action_scores(x,n_actions,hidden_state,action_idxs)
        # Clamp to the graph's schema count: a multi-domain model's
        # action_options can exceed it on schema-poor domains (Visitall: 1),
        # and the loop below indexes columns up to max_num_actions.
        self.max_num_actions = min(self.action_options, a_scores_new.shape[1])
        self.max_num_objects = self.object_options
        all_actions_batches,all_actions_scores = self.get_best_action_scores_locations(a_scores_new,self.max_num_actions)

        # Mixed-arity fix: per-action-index arity lookup, used to freeze the
        # scores of graphs whose action is already fully parameterized while
        # other graphs in the same beam still need more parameters.
        # (Comment this + the FIX lines below to restore old behavior.)
        arity_lookup = torch.tensor(
            [self.action_parameter_number_dict[i] for i in sorted(self.action_parameter_number_dict)],
            device=self.device)

        # Active beams now contain (token_ids, embeddings, scores)
        active_beams = []

        # Storage for completed sequences
        finished_beams = []
        finished_scores = []
        curr_depth = 0

        for action_idx in range(self.max_num_actions):
            decoder_input = self.get_best_action_embeddings(x,all_actions_batches[:,action_idx],n_node,domain_number_actions=self._decode_action_count(n_actions))
            #all_curr_action_scores = [elem[action_idx] for elem in all_actions_scores]
            #active_beams.append(([all_actions_batches[:,action_idx][0]], decoder_input,
            #                     all_curr_action_scores[0], hidden_state, curr_depth ))
            active_beams.append(([all_actions_batches[:,action_idx]], decoder_input,
                                 all_actions_scores[:,action_idx], hidden_state, curr_depth ))

        for parameter_number in range(self.max_number_action_parameters):  # replace with the actual maximum sequence length
            if not active_beams:
                break

            # Prepare batched inputs for the model
            current_decoder_input = torch.cat([beam[1] for beam in active_beams], dim=0)
            current_scores = [beam[2] for beam in active_beams]
            current_hidden = torch.cat([beam[3] for beam in active_beams], dim=1)
            curr_depth = active_beams[0][4]
            _, new_hidden = self.decoder(current_decoder_input, current_hidden)

            #new_hidden_split = torch.split(new_hidden, split_size_or_sections=1, dim=1)

            # Prepare for next step
            new_active_beams = []
            # compute_object_scores reads only .shape of this argument and
            # returns a fresh tensor, so allocating and zero-filling one here
            # was pure waste, once per parameter slot per decode. ao_scores has
            # the same shape; nothing reads its contents either.
            ao_scores_new = ao_scores

            # One row per graph for THIS parameter slot. Rows per graph are not
            # uniform, so the offset is the cumulative start of each graph's
            # rows (the arithmetic compute_object_scores uses) rather than a
            # fixed stride of max_number_action_parameters. A graph whose
            # action needs fewer parameters than this slot has no row here;
            # clamp into its own last row so the gather stays in bounds - its
            # score is frozen by done_prev below, so the value is never used.
            # Invariant across beams, so computed once here.
            parameter_locations = _par_starts + torch.minimum(
                torch.full_like(n_par_long, parameter_number), _par_last)

            for beam_idx, beam in enumerate(active_beams):
                ao_scores_new = self.compute_object_scores(x, n_parameters,n_objects,
                                                            ao_scores_new,
                                                            #new_hidden_split[beam_idx],
                                                            new_hidden[:, beam_idx*number_graphs :(beam_idx+1)*number_graphs],
                                                        object_idxs,parameter_number)
                all_objects_batches_all_params,all_objects_scores_all_params = self.get_best_action_object_scores_locations(
                                        ao_scores=ao_scores_new, n_node=n_node, k=self.max_num_objects,
                                        n_objects=n_objects, n_parameters=n_parameters)

                all_objects_batches = all_objects_batches_all_params[parameter_locations]
                all_objects_scores = all_objects_scores_all_params[parameter_locations]

                # Mixed-arity fix: which graphs in this beam are already done
                # (their action needed <= curr_depth params) - their scores are
                # frozen below. The finish check depends only on the beam's
                # actions, so it is hoisted out of the object loop.
                beam_actions = beam[0][0]
                beam_arities = arity_lookup[beam_actions]
                done_prev = beam_arities <= curr_depth
                beam_finished = bool((beam_arities <= curr_depth + 1).all())

                for object_option in range(self.max_num_objects):
                    all_objects = all_objects_batches[:,object_option]
                    all_curr_obj_scores = all_objects_scores[:,object_option]
                    new_decoder_input = self.get_best_object_embeddings_ltp(x,
                                                                            all_objects,
                                                                             n_node, number_graphs)


                    curr_sequence = active_beams[beam_idx][0] + [all_objects]
                    #action = curr_sequence[0].item()
                    actions = curr_sequence[0]
                    # Old version (updates every graph's score even after its
                    # action is fully parameterized - dilutes early finishers;
                    # finish check re-done per object option with .item() syncs):
                    #new_score = (current_scores[beam_idx]*(curr_depth+1) + all_curr_obj_scores)/ (curr_depth + 2 )
                    ##if curr_depth + 1 == self.action_parameter_number_dict[action]:
                    #if all([curr_depth + 1 >= self.action_parameter_number_dict[actions[idx].item()] for idx in range(actions.shape[0])]):
                    # Mixed-arity fix: freeze scores of already-done graphs
                    new_score_updated = (current_scores[beam_idx]*(curr_depth+1) + all_curr_obj_scores)/ (curr_depth + 2 )
                    new_score = torch.where(done_prev, current_scores[beam_idx], new_score_updated)
                    if beam_finished:
                        finished_beams.append(curr_sequence)
                        finished_scores.append(new_score)
                        continue

                    new_active_beams.append((curr_sequence, new_decoder_input,
                                             new_score,
                                             #new_hidden_split[beam_idx],
                                             new_hidden[:, beam_idx*number_graphs:(beam_idx+1)*number_graphs],
                                             curr_depth+1))
            
            active_beams = new_active_beams[:]

        ###
        ### PARALLELIZATION ALMOST DONE - JUST NEED to handle this last sorting
        ### currently, I just have total beams = num action * (num objects ^ num_params)
        ### Each beam has information about all input states in parallel
        ###
        # Sort using tensor operations instead of python lists 
        '''
        scores_tensor = torch.tensor(finished_scores, device=self.device)
        indices = torch.argsort(scores_tensor, descending=True)
        sorted_beams = [finished_beams[i] for i in indices.tolist()]
        sorted_scores = scores_tensor[indices].tolist()
        results = list(zip(sorted_scores, sorted_beams))
        return results
        '''
        return self.sort_parallel_beam_results(finished_scores, finished_beams)

    def sort_parallel_beam_results(self, finished_scores_parallel, finished_beams_parallel):
        """
        Sort parallel beam search results and return them in the same format as the original code.
        
        Args:
            finished_scores_parallel: List of tensors, each of shape [batch_size]
            finished_beams_parallel: List of beam sequences, where each sequence is a list of tensors
        
        Returns:
            List of results, where each result contains:
            - results[batch_idx]: List of (score, beam) tuples sorted by score in descending order
        """
        # Get batch size from the first score tensor
        batch_size = finished_scores_parallel[0].shape[0]
        
        # Initialize results for each batch
        all_results = []
        
        # Process each batch separately
        for batch_idx in range(batch_size):
            # Extract scores for this batch
            batch_scores = [score[batch_idx].item() for score in finished_scores_parallel]
            
            # Extract beams for this batch
            batch_beams = []
            for beam_seq in finished_beams_parallel:
                # Each beam_seq is a list of tensors where each tensor has shape [batch_size, ...]
                beam_for_batch = []
                for token_tensor in beam_seq:
                    # Extract the token(s) for this batch
                    if len(token_tensor.shape) == 1:
                        # Shape is [batch_size]
                        beam_for_batch.append(token_tensor[batch_idx])
                    else:
                        # Shape is [batch_size, 2] or similar
                        beam_for_batch.append(token_tensor[batch_idx])
                # Mixed-arity fix: graphs whose action finished before the rest
                # of the beam carry padding object tokens past their arity -
                # truncate to (action + its true parameter count).
                # (Comment the next two lines to restore old behavior.)
                arity = self.action_parameter_number_dict[int(beam_for_batch[0])]
                beam_for_batch = beam_for_batch[:1 + arity]
                batch_beams.append(beam_for_batch)

            # Convert scores to tensor for sorting
            scores_tensor = torch.tensor(batch_scores, device=finished_scores_parallel[0].device)

            # Sort indices in descending order
            indices = torch.argsort(scores_tensor, descending=True)

            # Create sorted beams and scores
            sorted_beams = [batch_beams[i] for i in indices.tolist()]
            sorted_scores = scores_tensor[indices].tolist()

            # Create results for this batch
            batch_results = list(zip(sorted_scores, sorted_beams))

            # Mixed-arity fix: truncation makes the padded continuations of an
            # early-finished graph identical - drop duplicates, keeping the
            # highest-scored occurrence (list is already sorted descending).
            # (Comment this block to restore old behavior.)
            seen_sequences = set()
            deduped_results = []
            for score, beam in batch_results:
                key = tuple(int(t) for t in beam)
                if key in seen_sequences:
                    continue
                seen_sequences.add(key)
                deduped_results.append((score, beam))
            batch_results = deduped_results

            all_results.append(batch_results)

        return all_results