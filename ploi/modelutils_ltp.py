import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from icecream import ic
from torch_geometric.nn import GATv2Conv,BatchNorm
from torch_geometric.data import DataLoader, HeteroData
from torch_geometric.nn import aggr
import time

class EdgeModelLtp(nn.Module):
    def __init__(self, n_features, n_edge_features, n_hidden, dropout=0.0):
        super(EdgeModelLtp, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_features + n_edge_features + n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )

    #def forward(self, out ):#src, dst, edge_attr, u=None, batch=None):
    def forward(self, src, dest, edge_attr, u=None, batch=None):
        # src, dst: [E, F_x], where E is num edges, F_x is node-feature dimensionality
        # edge_attr: [E, F_e], where E is num edges, F_e is edge-feature dimensionality
        #out = torch.cat([src, dst, edge_attr], dim=1)
        out = torch.cat([src,dest, edge_attr,u], dim=1)
        return self.edge_mlp(out)

class NodeModelLtp(nn.Module):
    def __init__(self, n_features, n_edge_features, n_hidden, n_targets, dropout=0.0):
        super(NodeModelLtp, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(n_features + n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(n_features + n_hidden*2, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )

    def forward(self, x, edge_idx, edge_attr, u=None, batch=None):
        row, col = edge_idx
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out,u], dim=1)
        #out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(nn.Module):
    def __init__(self, n_global_features,n_hidden, dropout=0.0):
        super(GlobalModel, self).__init__()
        self.global_mlp_1 = nn.Sequential(
            nn.Linear(n_global_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )
        self.global_mlp_2 = nn.Sequential(
            nn.Linear(n_hidden*3, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )
        self.aggr = aggr.SumAggregation()

    def forward(self, x, edge_index,edge_attr, u ,node_index):
        #self.global_edge_attention()
        #nodes_agg = torch.div(torch.mm(graph['nodes'].t(), (node_idxs == columns).float()).t(),len(node_idxs))
        #edges_agg = torch.div(torch.mm(graph['edges'].t(), (edge_idxs == columns).float()).t(),len(edge_idxs))

        #u = self.global_mlp_1(u)
        #nodes_agg = self.aggr(x,ptr=node_ptr)
        nodes_agg = self.aggr(x,index=node_index)
        edges_agg =self.aggr(edge_attr,index=edge_index) 

        #nodes_agg = torch.sum(x,dim=0).unsqueeze(0)
        #edges_agg = torch.sum(edge_attr,dim=0).unsqueeze(0)
        u = torch.cat([u, nodes_agg,edges_agg],dim=1)
        u = self.global_mlp_2(u)
        return u



class NodeUpdateAttn(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeUpdateAttn, self).__init__()
        #self.attention1 = GATv2Conv((-1, -1), hidden_channels, hidden_channels, add_self_loops=False)
        #self.attention2 = GATv2Conv((-1, -1), hidden_channels, out_channels, add_self_loops=False)
        self.attention1 = GATv2Conv((-1,-1), hidden_channels,heads=1, add_self_loops=False, edge_dim=hidden_channels)
        self.attention2 = GATv2Conv((-1,-1), hidden_channels,heads=1, add_self_loops=False, edge_dim=hidden_channels)
        
        self.lin_skip = nn.Linear(in_channels, out_channels)  # for the skip connection
        self.norm1 = nn.InstanceNorm1d(hidden_channels)
        self.norm2 = nn.InstanceNorm1d(out_channels)

    def forward(self, x, edge_index,edge_attr):
        # First attention layer
        h = F.relu(self.attention1((x,x), edge_index,edge_attr=edge_attr))
        h = self.norm1(h)
        
        # Second attention layer with skip connection
        h_out = F.relu(self.attention2((h, h), edge_index ,edge_attr=edge_attr) + self.lin_skip(x))
        h_out = self.norm2(h_out)

        return h_out


# Define the hetero graph neural network
class HeteroGNN(nn.Module):
    def __init__(self,n_features,n_edge_features,n_global_features,
                        representation_size ,dropout=None,num_rounds=3):
        super(HeteroGNN, self).__init__()
        
        # Node attention layers
        self.num_rounds = num_rounds
        self.representation_size = representation_size
        self.node_encoder = MLP([self.representation_size]*2,n_features)
        #self.node_update = NodeUpdateAttn(self.representation_size,self.representation_size,self.representation_size)
        self.node_update = NodeModelLtp(self.representation_size,self.representation_size,self.representation_size,self.representation_size)
        
        # Edge update linear layers
        #self.edge_encoder = nn.Linear(n_edge_features,self.representation_size)
        self.edge_encoder = MLP([self.representation_size]*2, n_edge_features)
        self.edge_update_network = EdgeModelLtp(self.representation_size,self.representation_size,self.representation_size)
        '''
        torch.nn.Sequential(
            torch.nn.Linear(2 * self.representation_size + self.representation_size, self.representation_size),
            torch.nn.ReLU(),
            nn.LayerNorm(self.representation_size),
            torch.nn.Linear(self.representation_size, self.representation_size)
        )
        '''
        self.global_encoder  = MLP([self.representation_size]*2,n_global_features)
        self.global_update = GlobalModel(self.representation_size,self.representation_size)
        
    def forward(self, batch):
        node_data  = batch['node'].x
        node_index = batch['node'].batch
        edge_index = torch.repeat_interleave(torch.arange(0, len(batch['globals'].x)).cuda(), batch['n_edge'].x)

        edge_features_node = batch['node','sends','node'].edge_attr
        edge_features_node_index = batch['node','sends','node'].edge_index
        global_data = batch['globals'].x
        
        node_data = self.node_encoder(node_data)
        edge_features_node = self.edge_encoder(edge_features_node)
        global_data = self.global_encoder(global_data)

        for i in range(self.num_rounds) :
            src = node_data[edge_features_node_index[0]]
            dest = node_data[edge_features_node_index[1]]
            global_node_repeat = global_data[node_index]
            global_edge_repeat = global_data[edge_index]

            #data['node'].x = F.relu(self.node_update((data, data, data['node', 'sends', 'node'].edge_index))
            edge_features_node = F.relu(self.edge_update_network(src,dest,edge_features_node,global_edge_repeat))
            node_data = F.relu(self.node_update(node_data,edge_features_node_index,edge_features_node,global_node_repeat))
            global_data = F.relu(self.global_update(node_data,edge_index,edge_features_node,global_data,node_index))
        return node_data, edge_features_node,global_data

class GNN_GRU(nn.Module):
    def __init__(self, n_features, n_edge_features,n_global_features,
                n_hidden, gnn_rounds,
                 num_decoder_layers,
                 dropout, 
                 action_space,
                 batch_size):
        #super().__init__(n_features, n_edge_features, n_hidden, dropout)
        super(GNN_GRU,self).__init__()
        #self.encoder = GraphNetworkLtp(n_features,n_edge_features,n_global_features\
        #                               ,n_hidden,dropout,gnn_rounds)

        self.encoder = HeteroGNN(n_features,n_edge_features,n_global_features\
                                       ,n_hidden,dropout,gnn_rounds)
        
        self.representation_size = n_hidden
        self.max_number_action_parameters = 0
        self.action_parameter_number_dict = {}
        #all_actions = [k for k, v in action_space.items()]
        self.number_actions = len(action_space.keys())
        self.action_options = 2
        self.object_options = 3
        #self.num_decoder_layers = num_decoder_layers
        number_graphs = batch_size

        if action_space != None:
            i = 0
            for key, values in action_space.items():
                self.max_number_action_parameters = max(len(values.params),\
                                                        self.max_number_action_parameters)
                self.action_parameter_number_dict[i] = len(values.params)
                i += 1

        #self.h0 = torch.zeros(num_decoder_layers,n_hidden).cuda()
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

    #def forward(self,x,edge_idx,edge_attr,u,a_scores, ao_scores, batch=None):
    def forward(self,data):
        number_graphs = data['n_node'].x.shape[0]
        h0 = torch.zeros(self.num_decoder_layers,number_graphs,self.representation_size).cuda()
        start_time = time.time()

        torch_time = time.time()
        action_idxs = torch.where(data['node'].x[:,0] == 1)[0]
        object_idxs = torch.where(data['node'].x[:,1] == 1)[0]

        a_scores = data['action_scores'].x
        ao_scores = data['action_object_scores'].x
        n_node = data['n_node'].x
        n_parameters = data['n_parameters'].x
        n_actions = data['n_action'].x
        n_objects = data['n_object'].x

        torch_where_time = time.time() - torch_time

        encoder_start_time = time.time()
        x,edge_attr, u = self.encoder(data)
        #data = self.encoder(data)
        encoder_time = time.time() - encoder_start_time

        decoder_time = time.time()
        u = u.unsqueeze(1)
        _,hidden_state = self.decoder(u,h0)
        decoder_total_time = time.time()-decoder_time

        all_actions_batches = []
        all_objects_batches = []
        all_actions_scores = []
        all_objects_scores = []
        #decoder_input = self.get_best_action_embedding()
        if self.training_mode :
            for i in range(a_scores.shape[0]):
                #all_actions_batches[0].append(torch.argmax(output_graph[-1]['action_scores'][i]))
                values , indexes = torch.topk(a_scores[i],1)
                all_actions_batches.append(indexes)
                all_actions_scores.append(values)
            all_actions = [elem[0] for elem in all_actions_batches]

            for i in range(ao_scores.shape[0]) :
            #for i in range(parameter_number, output_graph[-1]['action_object_scores'].shape[0],
            #               max_action_number_parameters):
                #values , indexes = torch.topk(output_graph[-1]['action_object_scores'][i],1)
                #values , indexes = torch.topk(output_graph[-1]['action_object_scores'][i][:max_objects_number],1)
                if len(n_node.shape) == 0:
                    current_graph_objects = int(n_node)
                else :
                    current_graph_objects = n_node[int(i/self.max_number_action_parameters)]

                values , indexes = torch.topk(ao_scores[i][:current_graph_objects],1)
                all_objects_batches.append(indexes)
                all_objects_scores.append(values)

            all_objects = [elem[0] for elem in all_objects_batches]

        x = self.action_score_decoder(x)

        action_scores_time = time.time()
        #action_scores = self.compute_action_scores(x,n_actions,hidden_state,action_idxs)
        a_scores = self.compute_action_scores(x,n_actions,hidden_state,action_idxs)
        action_scores_total_time = time.time() - action_scores_time

        #decoder_input = self.get_best_action_embedding(x,scores, all_actions,self.number_actions)
        computing_best_action_embedding = time.time()
        decoder_input = self.get_best_action_embeddings(x,all_actions,n_node,domain_number_actions=4)
        computing_best_action_embedding_time = time.time() - computing_best_action_embedding
        #ic (decoder_input.shape)

        decoder_time = time.time()
        action_object_scores_0 = []
        #ic(object_idxs.shape)
        #action_object_scores = torch.empty((1,object_idxs.shape[0])).cuda()
        #action_object_scores = torch.zeros(ao_scores.shape).cuda()
        #ic (action_object_scores.shape)
        for i in range(0, self.max_number_action_parameters):
            _, hidden_state = self.decoder(decoder_input, hidden_state)  # output:[32, 10004] [1, 32, 512] [32, 1, 27]

            #object_scores = torch.matmul (self.action_score_decoder(x[object_idxs]), hidden_state[-1])
            ao_scores += self.compute_object_scores(x, n_parameters,n_objects, ao_scores,hidden_state,
                                                       object_idxs,i)
            
            computing_best_object_embedding = time.time()
            if i == self.max_number_action_parameters-1 :
                break
            decoder_input = self.get_best_object_embeddings(x, all_objects, all_actions,parameter_number=i,
                                                            n_params=n_parameters,
                                                            n_node=n_node)    
            computing_best_object_embedding_time = time.time() - computing_best_object_embedding
        decoder_total_time_2 = time.time()-decoder_time
        return a_scores, ao_scores
        #return action_scores, action_object_scores

        #ic (action_scores,action_object_scores)

        end_time = time.time()
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

    def decode(self,x,edge_idx,edge_attr,u,batch=None):
        return 

    def greedy_decode(self, trg, decoder_hidden, encoder_outputs, ):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        seq_len, batch_size = trg.size()
        decoded_batch = torch.zeros((batch_size, seq_len))
        # decoder_input = torch.LongTensor([[EN.vocab.stoi['<sos>']] for _ in range(batch_size)]).cuda()
        decoder_input = Variable(trg.data[0, :]).cuda()  # sos
        print(decoder_input.shape)
        for t in range(seq_len):
            _ , hidden = self.decoder(decoder_input, hidden)
            topv, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi
            decoder_input = topi.detach().view(-1)

        return decoded_batch

    #def get_best_action_embedding(self,x,scores,all_actions,number_actions):
    #    return []

    def get_best_action_embeddings(self,x,all_actions,n_node,domain_number_actions):
        #if graph['nodes'].is_cuda:
        required_correct_features = torch.zeros((len(all_actions),1,self.representation_size),dtype=torch.float32).cuda()
        current_number_nodes = 0
        #ic (all_actions)
        for a,action in enumerate(all_actions) :
            action_curr_graph = n_node[a] - domain_number_actions
            #ic (graph['n_node'][a])
            required_correct_features[a][0] = x[current_number_nodes+action_curr_graph+action][:]

            current_number_nodes += n_node[a]
        return required_correct_features#,number_action_parameters

    def get_best_object_embeddings(self,x,all_objects,all_actions,parameter_number,n_params,n_node):
        current_number_nodes = 0
        #objects_counter = 0
        objects_counter = parameter_number
        feature_captured_object_counter = 0
        #ic (all_objects)
        #ic (parameter_number)
        #ic (graph['n_parameters'])

        required_correct_object_features = torch.zeros((len(all_actions), 1, self.representation_size),
                                                       dtype=torch.float32).cuda()
        #required_correct_object_features = required_correct_object_features.cuda()
        for a, action in enumerate(all_actions):

            object_idx = all_objects[objects_counter]
            required_correct_object_features[feature_captured_object_counter][0] = x[
                                                                      current_number_nodes + object_idx][:]

            objects_counter += int(n_params[a])

            feature_captured_object_counter += 1
            current_number_nodes += n_node[a]

        #ic (required_correct_object_features)
        return required_correct_object_features

    def compute_action_scores(self,x,n_actions,hidden_state, action_idxs):
        '''
        Computing the score for each of the actions (Updating the graph[action_scores]
        for all actions
            - each super graph has a certain number of action nodes (Number of graphs * num of actions per graph)
            - We score each set of actions in a graph (every diff value in global index refers to a diff graph)
            - Hence we compute scores for a set of each graph - over all actions and store it in action_scores (1 vector per graph)
        '''
        number_actions_array = [torch.tensor(0).cuda()]

        for elem in n_actions :
            number_actions_array.append(number_actions_array[-1]+elem)

        #ic (number_actions_array)
        #if graph['nodes'].is_cuda:
        #for i,elem in enumerate(number_actions_array):
        #    number_actions_array[i] = number_actions_array[i].cuda()

        return torch.stack([torch.matmul(x[action_idxs[int(number_actions_array[i]):int(number_actions_array[i + 1])]],hidden_state[-1,i]) for i in range(len(number_actions_array)-1)])

    #def compute_object_scores(self,graph, hidden_state,object_idxs,parameter_number):
    def compute_object_scores(self, x,n_params, n_objects , ao_scores,
                              hidden_state, object_idxs, parameter_number):
        number_objects_array = []
        #ic (object_idxs)
        '''
        if len(graph['n_object'].shape)== 0:
            elem_to_add = (0, graph['n_object'])
            #ic (graph['n_parameters'])
            for num_params in range(int(graph['n_parameters'])):
                number_objects_array.append(elem_to_add)
        else :
            #ic (graph['n_parameters'])
        '''
        #for i,num_objs in enumerate(graph['n_object']):
        for i,num_objs in enumerate(n_objects):
            if len(number_objects_array) == 0 :
                elem_to_add = (0,num_objs)
            else :
                elem_to_add = (number_objects_array[-1][1],number_objects_array[-1][1]+num_objs)

            for num_params in range(int(n_params[i])):
                number_objects_array.append(elem_to_add)
            #number_objects_array.append(number_objects_array[-1] + num_params)

        #ic (graph['nodes'].size)
        #ic (number_objects_array)
        #exit()
        #ic (hidden_state.shape)

        new_hidden_state = [None]*hidden_state.shape[0]

        #new_hidden_state = torch.repeat_interleave(hidden_state[0], new_tensor, dim=0)
        #ic (new_tensor)

        #if graph['nodes'].is_cuda::w

        '''
        #new_tensor = torch.tensor([int(elem[0]) for elem in n_params]).cuda()
        #new_tensor = new_tensor.long()
        '''
        #mask_matrix = torch.zeros(2, len(graph['action_object_scores']), 
        #                          len(graph['action_object_scores'][0])).cuda()
        mask_matrix = torch.zeros(2, ao_scores.shape[0], ao_scores.shape[1]).cuda()
        '''
            #new_tensor = new_tensor.cuda()
            #new_hidden_state = new_hidden_state.cuda()
            #mask_matrix = mask_matrix.cuda()
            #new_tensor = new_tensor.cuda()
            #for i,elem in enumerate(new_hidden_state):
            #    new_hidden_state[i] = new_hidden_state[i].cuda()
        else :
            new_tensor = torch.tensor([int(elem[0]) for elem in graph['n_parameters']])
            new_tensor = new_tensor.long()
            mask_matrix = torch.zeros(2, len(graph['action_object_scores']), len(graph['action_object_scores'][0]))
        '''

        for j in range(0,hidden_state.shape[0]):
            new_hidden_state[j] = torch.repeat_interleave(hidden_state[j],
                                                           n_params, dim=0)
        #ic(len(new_hidden_state))
        #ic(new_hidden_state[0].shape)
        #ic(new_hidden_state[-1].shape)
        #ic(new_hidden_state[-1][0].shape)
        #ic(new_hidden_state[-1][0])
        #ic (graph['action_object_scores'])
        #k = [torch.matmul(graph['nodes'][object_idxs[int(elem[0]):int(elem[1])]],
        #              new_hidden_state[i]) for i,elem in enumerate(number_objects_array)]
        current_parameter_indexes = []
        action_object_score_counter = 0
        '''
        for i in graph['n_parameters']:
            #ic (i,action_object_score_counter)
            #ic (parameter_number)
            if parameter_number == 0:
                current_parameter_indexes.append(int(action_object_score_counter))
            elif parameter_number == 1 :
                if i == 2 :
                    current_parameter_indexes.append(int(action_object_score_counter)+i-1)
            action_object_score_counter += i
        '''
        #ic (graph['action_object_scores'])
        for i in n_params:
            #current_parameter_indexes.append(torch.tensor(action_object_score_counter+parameter_number))
            current_parameter_indexes.append(action_object_score_counter+parameter_number)
            action_object_score_counter += i.item()
        #ic (current_parameter_indexes)
        #exit()

        #if graph['nodes'].is_cuda :
        #mask_matrix[0, current_parameter_indexes, :] = torch.ones(mask_matrix[0,current_parameter_indexes,:].shape).cuda()
        #mask_matrix[1, list(set([i for i in range(len(graph['action_object_scores']))]) - set(current_parameter_indexes)), :] = torch.ones(mask_matrix[1, list(set([i for i in range(len(graph['action_object_scores']))]) - set(current_parameter_indexes)), :].shape).cuda()
        mask_matrix[0, current_parameter_indexes, :] = torch.ones(mask_matrix[0,current_parameter_indexes,:].shape).cuda()
        mask_matrix[1, list(set([i for i in range(ao_scores.shape[0])]) - set(current_parameter_indexes)), :] = torch.ones(mask_matrix[1, list(set([i for i in range(ao_scores.shape[0])]) - set(current_parameter_indexes)), :].shape).cuda()

        '''
        else :
            mask_matrix[0, current_parameter_indexes, :] = 1
            mask_matrix[1, list(set([i for i in range(len(graph['action_object_scores']))]) - set(current_parameter_indexes)), :] = 1
        '''
        #ic (graph['n_object'])
        #ic ([torch.matmul(graph['nodes'][object_idxs[int(elem[0]):int(elem[1])]],
        #                  new_hidden_state[i]) for i,elem in enumerate(number_objects_array)])
        #ic ([torch.stack([torch.matmul(graph['nodes'][object_idxs[int(elem[0]):int(elem[1])]],new_hidden_state[j][i]) for j in range(0,hidden_state.shape[0])],dim=0).sum(dim=0) for i,elem in enumerate(number_objects_array)])
        #j=0
        #ic ([torch.matmul(graph['nodes'][object_idxs[int(elem[0]):int(elem[1])]],
        #                  new_hidden_state[i]) for i,elem in enumerate(number_objects_array)])
        #j=1
        #ic ([torch.matmul(graph['nodes'][object_idxs[int(elem[0]):int(elem[1])]],
        #                                     new_hidden_state[i]) for i,elem in enumerate(number_objects_array)])
        #exit()
        #updated_action_scores =[torch.matmul(graph['nodes'][object_idxs[int(elem[0]):int(elem[1])]],
        #                                     new_hidden_state[i]) for i,elem in enumerate(number_objects_array)]
        #updated_action_scores =[torch.matmul(x[object_idxs[int(elem[0]):int(elem[1])]],
        #                                     new_hidden_state[-1][i]) for i,elem in enumerate(number_objects_array)]
        updated_action_scores =[torch.matmul(x[object_idxs[int(elem[0]):int(elem[1])]],
                                             new_hidden_state[-1][i]) for i,elem in enumerate(number_objects_array)]
        #updated_action_scores  = [torch.stack([torch.matmul(graph['nodes'][object_idxs[int(elem[0]):int(elem[1])]],new_hidden_state[j][i]) for j in range(0,hidden_state.shape[0])],dim=0).sum(dim=0) for i,elem in enumerate(number_objects_array)]
        #ic (updated_action_scores)

        #max_length = len(graph['action_object_scores'][0])
        #ic (graph['action_object_scores'])

        
        max_length = ao_scores.shape[1]
        #ic (max_length)
        variable_action_object_scores = []
        for elem in updated_action_scores:
            variable_action_object_scores.append(F.pad(elem, (0,max_length-elem.shape[0]), "constant", 0))

        variable_action_object_scores = torch.stack(variable_action_object_scores)
        #if graph['nodes'].is_cuda:
        variable_action_object_scores = variable_action_object_scores.cuda()
        
        #ic (variable_action_object_scores)

        #ic (variable_action_object_scores)
        #ic (updated_action_scores)
        '''
        replacements = {
        'action_object_scores': torch.mul(mask_matrix[0], variable_action_object_scores) + torch.mul(mask_matrix[1], graph['action_object_scores'])
        }
        '''
        #new_ao_scores = torch.mul(mask_matrix[0], variable_action_object_scores) + torch.mul(mask_matrix[1], ao_scores)
        #new_ao_scores = torch.mul(mask_matrix[0], variable_action_object_scores) + torch.mul(mask_matrix[1], ao_scores)

        return torch.mul(mask_matrix[0],variable_action_object_scores)
        #return torch.mul(mask_matrix[0], variable_action_object_scores) + torch.mul(mask_matrix[1], ao_scores)
        #ic (replacements['action_object_scores'])
        #exit()
        #ic (mask_matrix.requires_grad)
        #ic (new_tensor.requires_grad)
        #del mask_matrix
        #del variable_action_object_scores
        #del new_tensor

        #return new_ao_scores
        #return replace(graph,replacements)


def MLP(layers, input_dim, dropout=0.):
    """Create MLP
    """
    mlp_layers = [nn.Linear(input_dim, layers[0])]

    for layer_num in range(0, len(layers)-1):
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(layers[layer_num], layers[layer_num+1]))
    if len(layers) > 1:
        mlp_layers.append(nn.LayerNorm(mlp_layers[-1].weight.size()[:-1]))
        if dropout > 0:
            mlp_layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*mlp_layers)


class GraphNetworkLtp(nn.Module):
    def __init__(self, n_features, n_edge_features,n_global_features, n_hidden, dropout=0.0,gnn_rounds=3):
        super(GraphNetworkLtp, self).__init__()

        self.meta_layers = []

        self.meta_layer_1 = self.build_meta_layer(
            n_features, n_edge_features, n_global_features,n_hidden, n_hidden, dropout=dropout
        )
        self.meta_layers.append(self.meta_layer_1)

        for i in range(1,gnn_rounds):
            self.meta_layers.append(self.build_meta_layer(
                n_hidden, n_hidden, n_hidden,n_hidden, n_hidden, dropout=dropout
            )
        )
        self.meta_layers = nn.ModuleList(self.meta_layers)
        self.gnn_rounds = gnn_rounds

    def build_meta_layer(
        self, n_features, n_edge_features,n_global_features, n_hidden, n_targets, dropout=0.0
    ):
        return MetaLayer(
            edge_model=EdgeModelLtp(
                n_features, n_edge_features, n_hidden, dropout=dropout
            ),
            node_model=NodeModelLtp(
                n_features, n_edge_features, n_hidden, n_targets, dropout=dropout
            ),
            global_model=GlobalModel(
                n_global_features, n_hidden, dropout=dropout
            ),
        )

    #def forward(self, x, edge_idx, edge_attr, u=None, batch=None):
    def forward(self, data, u=None, batch=None):
        x = data.x
        edge_idx = data.edge_index
        edge_attr = data.edge_attr
        for idx in range (self.gnn_rounds):
            x, edge_attr, u = self.meta_layers[idx](x, edge_idx, edge_attr,u)
        #ic (x.shape)
        #ic (x.shape)
        #ic (edge_attr.shape)
        #ic (u.shape)
        return x,edge_attr,u