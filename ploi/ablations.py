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
import itertools
import time
from ploi.modelutils_ltp import GNN_GRU, HeteroGNN_global, HeteroGNN ,EncodeDecode

class GNN_non_CD_decode(EncodeDecode):
    def __init__(self, n_features, n_edge_features,n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 attn_dropout ,
                 action_space,
                 batch_size,
                 n_heads,
                 g_node,
                 device,
                 action_options,
                 object_options):
        super(GNN_non_CD_decode,self).__init__(
            n_features, n_edge_features, n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 attn_dropout ,
                 action_space,
                 batch_size,
                 n_heads,
                 g_node,
                 device,
                 action_options,
                 object_options,
        )
        self.max_num_actions = 1
        self.max_num_objects = 1
        self.device = device

        if g_node is True :
            self.encoder = HeteroGNN_global(n_features,n_edge_features,n_global_features\
                                        ,n_hidden,dropout,attn_dropout,gnn_rounds,n_heads,device)
        else :
            self.encoder = HeteroGNN(n_features,n_edge_features,n_global_features\
                                        ,n_hidden,dropout,attn_dropout,gnn_rounds,n_heads,device)
        
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

        #self.decoder = nn.GRU(n_hidden, hidden_size=n_hidden,\
        #                      num_layers=num_decoder_layers,bias=False,batch_first=True)

        #self.num_decoder_layers = num_decoder_layers
        self.action_score_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            #nn.Linear(n_hidden, self.number_actions),
            nn.Linear(n_hidden, n_hidden),
        )

        self.object_score_decoder = nn.Sequential(
            nn.Linear(n_hidden + self.max_number_action_parameters, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )
        self.training_mode = True

    def forward(self,data, beam_search = False):
        start_time = time.time()

        graph_info = self.extract_graph_info_ltp(data)
        action_idxs, object_idxs, a_scores, ao_scores, n_node, n_parameters, n_actions, n_objects,number_graphs = graph_info 

        encoder_start_time = time.time()
        x,edge_attr, u = self.encoder(data)
        encoder_time = time.time() - encoder_start_time

        decoder_time = time.time()
        updated_global = self.action_score_decoder(u).unsqueeze(0)

        action_scores_time = time.time()
        a_scores_new = self.compute_action_scores(x,n_actions, updated_global,action_idxs)

        if beam_search == False :
            return self.non_beam_decode_non_CD(x, u ,a_scores_new,ao_scores,n_node,
                                        n_parameters,n_objects,
                                        object_idxs,n_actions,action_idxs, number_graphs)
    
        else :
            return self.beam_search_non_CD(x,u ,number_graphs=number_graphs,
                         ao_scores=ao_scores,a_scores_new=a_scores_new,n_node=n_node,n_parameters=n_parameters,
                         n_objects=n_objects,object_idxs=object_idxs,n_actions=n_actions,
                         action_idxs=action_idxs)
            
class GNN_non_AG_CD(EncodeDecode):
    def __init__(self, n_features, n_edge_features,n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 attn_dropout ,
                 action_space,
                 batch_size,
                 n_heads,
                 g_node,
                 device,
                 action_options,
                 object_options):
        super(GNN_non_AG_CD,self).__init__(
            n_features, n_edge_features, n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 attn_dropout ,
                 action_space,
                 batch_size,
                 n_heads,
                 g_node,
                 device,
                 action_options,
                 object_options,
        )
        self.max_num_actions = 1
        self.max_num_objects = 1
        self.device = device

        if g_node is True :
            self.encoder = HeteroGNN_global(n_features,n_edge_features,n_global_features\
                                        ,n_hidden,dropout,attn_dropout,gnn_rounds,n_heads,device)
        else :
            self.encoder = HeteroGNN(n_features,n_edge_features,n_global_features\
                                        ,n_hidden,dropout,attn_dropout,gnn_rounds,n_heads,device)
        
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

        self.decoder = nn.GRU(n_hidden + self.number_actions, hidden_size=n_hidden,\
                              num_layers=num_decoder_layers,bias=False,batch_first=True)

        self.num_decoder_layers = num_decoder_layers
        self.action_score_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.number_actions),
        )

        self.object_score_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )
        self.training_mode = True

    def forward(self,data, beam_search = False):
        start_time = time.time()

        graph_info = self.extract_graph_info_ltp(data)
        action_idxs, object_idxs, a_scores, ao_scores, n_node, n_parameters, n_actions, n_objects,number_graphs = graph_info 
        #h0 = torch.zeros(self.num_decoder_layers,number_graphs,self.representation_size,device=self.device)

        encoder_start_time = time.time()
        x,edge_attr, u = self.encoder(data)
        encoder_time = time.time() - encoder_start_time

        decoder_time = time.time()

        if beam_search == False :
            return self.non_beam_decode(x,u,a_scores,ao_scores,n_node,
                                        n_parameters,n_objects,object_idxs,
                                        n_actions,action_idxs,number_graphs)
    
        else :
            return self.beam_search_v2(x,u ,number_graphs=number_graphs,
                         ao_scores=ao_scores,n_node=n_node,n_parameters=n_parameters,
                         n_objects=n_objects,object_idxs=object_idxs,n_actions=n_actions,
                         action_idxs=action_idxs)

    def non_beam_decode(self,x,u ,a_scores,
                        ao_scores,n_node,n_parameters,
                        n_objects,object_idxs,n_actions,
                        action_idxs,number_graphs):
        a_scores_new = self.action_score_decoder(u)
        ao_scores_new = torch.zeros(ao_scores.shape,device=self.device)
        all_actions_batches,_ = self.get_best_action_scores_locations(a_scores,self.max_num_actions)
        all_objects_batches,_ = self.get_best_action_object_scores_locations(ao_scores,n_node,self.max_num_objects)

        hidden_state = torch.zeros(self.num_decoder_layers,number_graphs,self.representation_size,device=self.device)
        decoder_input = torch.cat([u,a_scores], dim=1).unsqueeze(1)
        
        #if self.training_mode :
        all_actions = [elem[0] for elem in all_actions_batches]
        all_objects = [elem[0] for elem in all_objects_batches]

        for i in range(0, self.max_number_action_parameters):
            _, hidden_state = self.decoder(decoder_input, hidden_state) 
            ao_scores_new += self.compute_object_scores(x, n_parameters,n_objects, ao_scores,hidden_state,
                                                       object_idxs,i)
            computing_best_object_embedding = time.time()
            if i == self.max_number_action_parameters-1 :
                break
            obj_decoder_input = self.get_best_object_embeddings(x, all_objects, all_actions,parameter_number=i,
                                                            n_params=n_parameters,
                                                            n_node=n_node)    
            decoder_input = torch.cat([obj_decoder_input,a_scores.unsqueeze(1)],dim=2 )
            computing_best_object_embedding_time = time.time() - computing_best_object_embedding

        return a_scores_new, ao_scores_new

    def beam_search_v2(self, x, u, number_graphs,ao_scores,n_node,
                    n_parameters,n_objects,object_idxs,n_actions,
                    action_idxs):

        #a_scores_new = self.compute_action_scores(x,n_actions,hidden_state,action_idxs)
        a_scores_new = self.action_score_decoder(u)
        self.max_num_actions = self.action_options
        self.max_num_objects = self.object_options
        all_actions_batches,all_actions_scores = self.get_best_action_scores_locations(a_scores_new,self.max_num_actions)

        hidden_state = torch.zeros(self.num_decoder_layers,number_graphs,self.representation_size,device=self.device)

        # Active beams now contain (token_ids, embeddings, scores)
        active_beams = []
        
        # Storage for completed sequences
        finished_beams = []
        finished_scores = []
        curr_depth = 0

        top_action_scores, top_action_indices = torch.topk(a_scores_new,self.max_num_actions)

        for action_idx in range(self.max_num_actions):
            action_one_hot = F.one_hot(torch.tensor(top_action_indices[0][action_idx], device=self.device), num_classes=self.number_actions).unsqueeze(0) 
            decoder_input = torch.cat([u,action_one_hot], dim=1).unsqueeze(1)
            active_beams.append(([top_action_indices[0][action_idx]], decoder_input, 
                                 top_action_scores[0][action_idx], hidden_state , curr_depth,
                                  action_one_hot ))

        for parameter_number in range(self.max_number_action_parameters):  # replace with the actual maximum sequence length
            if not active_beams:
                break

            # Prepare batched inputs for the model
            current_decoder_input = torch.cat([beam[1] for beam in active_beams], dim=0)
            current_scores = [beam[2] for beam in active_beams]
            current_hidden = torch.cat([beam[3] for beam in active_beams], dim=1)
            curr_depth = active_beams[0][4]
            _, new_hidden = self.decoder(current_decoder_input, current_hidden)

            new_hidden_split = torch.split(new_hidden, split_size_or_sections=1, dim=1)

            # Prepare for next step
            new_active_beams = []
            #ao_scores_new = torch.zeros(ao_scores.shape).cuda()
            ao_scores_new = torch.zeros(ao_scores.shape,device=self.device)

            for beam_idx, beam in enumerate(active_beams):
                ao_scores_new = self.compute_object_scores(x, n_parameters,n_objects, 
                                                            ao_scores_new,new_hidden_split[beam_idx],
                                                        object_idxs,parameter_number)
                all_objects_batches_all_params,all_objects_scores_all_params = self.get_best_action_object_scores_locations(
                                        ao_scores=ao_scores_new, n_node=n_node, k=self.max_num_objects)

                action_one_hot = beam[5]
                all_objects_scores = all_objects_scores_all_params[parameter_number]
                all_objects_batches = all_objects_batches_all_params[parameter_number]
                for object_option in range(self.max_num_objects):
                    all_objects = [all_objects_batches[object_option]] 
                    all_curr_obj_scores = [all_objects_scores[object_option]]
                    new_decoder_input_obj = self.get_best_object_embeddings_ltp(x, all_objects, n_node, number_graphs)
                    new_decoder_input = torch.cat([new_decoder_input_obj, action_one_hot.unsqueeze(1)], dim=2)

                    curr_sequence = active_beams[beam_idx][0] + [all_objects[0]] 

                    action = curr_sequence[0].item()
                    new_score = (current_scores[beam_idx]*(curr_depth+1) + all_curr_obj_scores[0])/ (curr_depth + 2 )
                    if curr_depth + 1 == self.action_parameter_number_dict[action]:
                        finished_beams.append(curr_sequence)
                        finished_scores.append(new_score)
                        continue

                    new_active_beams.append((curr_sequence, new_decoder_input, 
                                             new_score, new_hidden_split[beam_idx],curr_depth+1,
                                             action_one_hot))
            
            active_beams = new_active_beams[:]

        results = sorted(zip(finished_scores, finished_beams), reverse=True)
        return results

class GNN_non_AG_non_CD(EncodeDecode):
    def __init__(self, n_features, n_edge_features,n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 attn_dropout ,
                 action_space,
                 batch_size,
                 n_heads,
                 g_node,
                 device,
                 action_options,
                 object_options):
        super(GNN_non_AG_non_CD,self).__init__(
            n_features, n_edge_features, n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 attn_dropout ,
                 action_space,
                 batch_size,
                 n_heads,
                 g_node,
                 device,
                 action_options,
                 object_options,
        )
        self.max_num_actions = 1
        self.max_num_objects = 1
        self.device = device

        if g_node is True :
            self.encoder = HeteroGNN_global(n_features,n_edge_features,n_global_features\
                                        ,n_hidden,dropout,attn_dropout,gnn_rounds,n_heads,device)
        else :
            self.encoder = HeteroGNN(n_features,n_edge_features,n_global_features\
                                        ,n_hidden,dropout,attn_dropout,gnn_rounds,n_heads,device)
        
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

        #self.decoder = nn.GRU(n_hidden, hidden_size=n_hidden,\
        #                      num_layers=num_decoder_layers,bias=False,batch_first=True)

        #self.num_decoder_layers = num_decoder_layers
        self.action_score_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.number_actions),
            #nn.Linear(n_hidden, n_hidden),
        )

        self.object_score_decoder = nn.Sequential(
            nn.Linear(n_hidden + self.max_number_action_parameters, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )
        self.training_mode = True

    def forward(self,data, beam_search = False):
        start_time = time.time()

        graph_info = self.extract_graph_info_ltp(data)
        action_idxs, object_idxs, a_scores, ao_scores, n_node, n_parameters, n_actions, n_objects,number_graphs = graph_info 

        encoder_start_time = time.time()
        x,edge_attr, u = self.encoder(data)
        encoder_time = time.time() - encoder_start_time

        decoder_time = time.time()
        #updated_global = self.action_score_decoder(u).unsqueeze(0)
        a_scores_new = self.action_score_decoder(u)

        action_scores_time = time.time()
        #a_scores_new = self.compute_action_scores(x,n_actions, updated_global,action_idxs)

        if beam_search == False :
            return self.non_beam_decode_non_CD(x, u ,a_scores_new,ao_scores,n_node,
                                        n_parameters,n_objects,
                                        object_idxs,n_actions,action_idxs, number_graphs)
    
        else :
            return self.beam_search_non_CD(x,u ,number_graphs=number_graphs,
                         ao_scores=ao_scores,a_scores_new=a_scores_new,n_node=n_node,n_parameters=n_parameters,
                         n_objects=n_objects,object_idxs=object_idxs,n_actions=n_actions,
                         action_idxs=action_idxs)
            
