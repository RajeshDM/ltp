
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_softmax
from torch_geometric.nn import aggr
from typing import Dict, Any, Tuple, Optional, List, Union
from ploi.attention_layer import MLP, GraphAttentionV2Layer

class EdgeModelLtp_Scriptable(nn.Module):
    def __init__(self, n_features: int, n_edge_features: int, n_global: int, n_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_features + n_edge_features + n_global, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )

    def forward(self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor, 
                u: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u is not None:
            out = torch.cat([src, dest, edge_attr, u], dim=1)
        else:
            out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class GlobalModel_Scriptable(nn.Module):
    def __init__(self, n_global_features: int, n_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.global_mlp_2 = nn.Sequential(
            nn.Linear(n_global_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
        )
        self.aggr = aggr.SumAggregation()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, u: torch.Tensor, 
                node_index: torch.Tensor) -> torch.Tensor:
        nodes_agg = self.aggr(x, index=node_index)
        edges_agg = self.aggr(edge_attr, index=edge_index)
        u_cat = torch.cat([u, nodes_agg, edges_agg], dim=1)
        return self.global_mlp_2(u_cat)

class HeteroGNN_global_Scriptable(nn.Module):
    def __init__(self, n_features: int, n_edge_features: int, n_global_features: int,
                 representation_size: int, dropout: float = 0.0, attn_dropout: float = 0.0,
                 num_rounds: int = 3, n_heads: int = 1, device: str = 'cuda:0'):
        super().__init__()
        
        self.device = device
        self.num_rounds = num_rounds
        self.representation_size = representation_size

        # Node related layers
        self.node_encoder = MLP([self.representation_size]*2, n_features)
        self.node_attention_layer = GraphAttentionV2Layer(
            in_features_1=self.representation_size*2,
            out_features_1=self.representation_size*2,
            in_features_2=self.representation_size,
            out_features_2=self.representation_size,
            n_heads=n_heads,
            is_concat=False,
            dropout=attn_dropout,
            leaky_relu_negative_slope=0.2,
            share_weights=False
        )
        
        self.node_update = MLP([self.representation_size]*2, self.representation_size*5, dropout=dropout)
        
        # Edge related layers
        self.edge_encoder = MLP([self.representation_size]*2, n_edge_features)
        self.edge_update = EdgeModelLtp_Scriptable(
            self.representation_size*2,  # Node features
            self.representation_size*2,  # Edge features
            self.representation_size*2,  # Global features
            self.representation_size,    # Hidden features
            dropout=dropout
        )

        # Global related layers
        self.global_encoder = MLP([self.representation_size]*2, n_global_features)
        self.global_update = GlobalModel_Scriptable(self.representation_size*4, self.representation_size, dropout=dropout)
        
    def forward(self, node_data: torch.Tensor, node_index: torch.Tensor, edge_index: torch.Tensor,
                edge_features_node: torch.Tensor, edge_features_node_index: torch.Tensor,
                global_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Extract source and destination indices
        src_indices = edge_features_node_index[0]
        dest_indices = edge_features_node_index[1]
        
        # Encode features
        node_data = self.node_encoder(node_data)
        edge_features_node = self.edge_encoder(edge_features_node)
        global_data = self.global_encoder(global_data)

        # Store original encoded features
        node_data_original = node_data
        edge_features_node_original = edge_features_node
        global_data_original = global_data

        # Message passing rounds
        for i in range(self.num_rounds):
            # Concatenate original and current features
            node_data = torch.cat([node_data_original, node_data], dim=1)
            edge_features_node = torch.cat([edge_features_node_original, edge_features_node], dim=1)
            global_data = torch.cat([global_data_original, global_data], dim=1)

            # Gather source and destination node features
            src = node_data[src_indices]
            dest = node_data[dest_indices]
            
            # Replicate global features for nodes and edges
            global_node_repeat = global_data[node_index]
            global_edge_repeat = global_data[edge_index]

            # Update edge features
            edge_features_node = self.edge_update(src, dest, edge_features_node, global_edge_repeat)
            
            # Update node features using attention
            node_data = self.node_update(
                self.node_attention_layer(node_data, edge_features_node, dest_indices, global_node_repeat)
            )
            
            # Update global features
            global_data = self.global_update(node_data, edge_index, edge_features_node, global_data, node_index)
            
        return node_data, edge_features_node, global_data


class HeteroGNN_global_Wrapper(nn.Module):
    def __init__(self, n_features: int, n_edge_features: int, n_global_features: int,
                 representation_size: int, dropout: float = 0.0, attn_dropout: float = 0.0,
                 num_rounds: int = 3, n_heads: int = 1, device: str = 'cuda:0'):
        super().__init__()
        
        self.device = device
        
        # Create the scriptable model
        self.model = HeteroGNN_global_Scriptable(
            n_features=n_features,
            n_edge_features=n_edge_features,
            n_global_features=n_global_features,
            representation_size=representation_size,
            dropout=dropout,
            attn_dropout=attn_dropout,
            num_rounds=num_rounds,
            n_heads=n_heads,
            device=device
        )
    
    def forward(self, batch):
        # Extract data from HeteroDataBatch
        node_data = batch['node'].x
        node_index = batch['node'].batch
        
        # Create edge index for global features
        edge_index = torch.repeat_interleave(
            torch.arange(0, len(batch['globals'].x), device=self.device), 
            batch['n_edge'].x
        )

        # Extract edge features
        edge_features_node = batch['node', 'sends', 'node'].edge_attr
        edge_features_node_index = batch['node', 'sends', 'node'].edge_index
        global_data = batch['globals'].x
        
        # Forward pass through the scriptable model
        return self.model(
            node_data, 
            node_index, 
            edge_index,
            edge_features_node, 
            edge_features_node_index, 
            global_data
        )