import torch
import torch.nn as nn
from icecream import ic
from torch_geometric.nn import MessagePassing
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter
import torch.nn.functional as F
import time
from torch_scatter import scatter_max
from typing import Optional, Tuple, Dict, Any, List

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

    #def forward(self, h, e, receivers,u):
    def forward(self, h: torch.Tensor, e: torch.Tensor, receivers: torch.Tensor, 
            u: Optional[torch.Tensor] = None) -> torch.Tensor:
        #,h: torch.Tensor,e:torch.Tensor, adj_mat: torch.Tensor,senders,receivers):
        """
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`
        """
        # Number of nodes
        n_nodes = h.size(0)
        n_edges = e.size(0)
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden_1)
        g_r = self.linear_r(e).view(n_edges, self.n_heads, self.n_hidden_2)
        receiver_counts = torch.bincount(receivers, minlength=n_nodes)
        '''
        unique_info = torch.unique(receivers, return_counts=True, sorted=True,return_inverse=True)
        receiver_counts = torch.zeros(n_nodes,dtype=torch.long,device=g_l.device)
        receiver_counts[unique_info[0]] = unique_info[2]
        '''
        g_concat = torch.cat((torch.repeat_interleave(g_l,receiver_counts,dim=0), g_r), dim=2)

        attn_softmax = scatter_softmax(self.dropout(self.attn(self.activation(g_concat))), receivers, dim=0)

        # Apply softmax weights to features and aggregate
        aggregated_effects = scatter(
            g_r * attn_softmax,
            receivers,
            dim=0,
            dim_size=n_nodes,
            reduce='sum'
        )

        if u is not None:
            out = torch.cat([g_l.mean(dim=1),aggregated_effects.mean(dim=1),u],dim=1)
        else :
            out = torch.cat([g_l.mean(dim=1),aggregated_effects.mean(dim=1)],dim=1)

        return out