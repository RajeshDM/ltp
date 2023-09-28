import torch
import torch.nn as nn
from icecream import ic
from torch_geometric.nn import MessagePassing
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter
import torch.nn.functional as F
import time

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

def prepare_adjacency_matrix(num_nodes,receivers):
    #num_nodes = graph['nodes'].size()[0]
    #num_edges = graph['edges'].size()[0]

    columns = torch.arange(0, num_nodes).long().cuda()
    #columns = columns.cuda()
    #ic (graph['senders'])
    #ic (columns)
    #ic (graph[edge_end_string].view(-1))
    #ic (columns)
    #ic (graph['receivers'])
    #ic (graph[edge_end_string].view(-1)[:,None])
    rec_m = receivers.view(-1)[:,None] == columns
    #ic (rec_m)
    #for elem in rec_m:
    #    print (elem)
    return rec_m.float()

class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features_1 : int, out_features_1: int,
                in_features_2 : int, out_features_2: int,
                n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features_1 % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden_1 = out_features_1 // n_heads
            self.n_hidden_2 = out_features_2 // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden_1 = out_features_1
            self.n_hidden_2 = out_features_2

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features_1, self.n_hidden_1 * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features_2, self.n_hidden_2 * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        # Instead of the addition, we are doing concatenation because of the node-edge asymmetry
        self.attn = nn.Linear(self.n_hidden_1+self.n_hidden_2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

        self.node_update = MLP([self.n_hidden_1]*2,self.n_hidden_1*3) 


    def forward(self, h, e, receivers,u):
        #,h: torch.Tensor,e:torch.Tensor, adj_mat: torch.Tensor,senders,receivers):
        """
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`
        """
        # Number of nodes
        n_nodes = h.shape[0]
        n_edges = e.shape[0]
        #ic (self.n_hidden)
        #start_time = time.time()
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden_1)
        #g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(e).view(n_edges, self.n_heads, self.n_hidden_2)
        #linear_time = time.time()
        #ic ("linear time", linear_time-start_time)
        #ic (g_r.shape)
        #g_r_repeat_interleave = g_r.repeat_interleave(n_edges, dim=0)
        unique_info = torch.unique(receivers, return_counts=True, sorted=True,return_inverse=True)
        receiver_counts = torch.zeros(n_nodes,dtype=torch.long).cuda()
        receiver_counts[unique_info[0]] = unique_info[2]

        g_l_repeat = torch.repeat_interleave(g_l,receiver_counts,dim=0)
        g_concat = torch.cat((g_l_repeat, g_r),dim=2)

        # Calculate
        attn = self.attn(self.activation(g_concat))
        attn = self.dropout(attn)
        attn_softmax = scatter_softmax(attn, receivers, dim=0)

        g_r_with_attn = g_r * attn_softmax

        #aggregated_effects = torch.zeros((n_nodes,self.n_heads,self.n_hidden_1)).cuda()
        #aggregated_effects[torch.arange(torch.max(receivers)+1)] = scatter(g_r_with_attn, receivers, dim=0, reduce='add')
        aggregated_effects = torch.zeros(h.shape).cuda()
        aggregated_effects[torch.arange(torch.max(receivers)+1)] = scatter(g_r_with_attn.squeeze(), receivers, dim=0, reduce='add')

        #ic (g_l.squeeze(1).shape)
        out = torch.cat([g_l.squeeze(1),aggregated_effects,u],dim=1)
        out = self.node_update (out)
        return out

    def forward_old(self, h,g_l, e, receivers,u):
        e = e.squeeze(-1)
        rec_m = prepare_adjacency_matrix(n_nodes,receivers)
        count = 0
        number_zero = 0
        updating_rec_time_start = time.time()
        for i,curr_count in enumerate(receiver_counts):
            #node_indice = unique_info[0][i]
            if curr_count == 0 :
                number_zero += 1
                continue
            edge_indices = (unique_info[1] == i-number_zero).nonzero(as_tuple=True)[0]
            rec_m[edge_indices,i] = self.softmax(e[count:count+curr_count]).squeeze(1)
            count = count + curr_count
        aggregated_effects = torch.mm(g_r.squeeze(1).t(), rec_m)
        return torch.cat((g_l.squeeze(1),aggregated_effects.t(),u),dim=1)