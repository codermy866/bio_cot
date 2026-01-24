import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DynamicClinicalGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Learnable similarity metric for edge weights
        self.similarity_net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Graph Attention Network (GAT) layers
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        
        # Global attention
        self.global_attention = nn.MultiheadAttention(hidden_dim * num_heads, num_heads)
        
        # Temporal dynamics
        self.temporal_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final projection
        self.fc = nn.Linear(hidden_dim * num_heads, out_dim)

    def construct_dynamic_graph(self, meta, temporal):
        batch_size = meta.size(0)
        device = meta.device
        
        # Compute pairwise metadata similarity
        meta_expanded = meta.unsqueeze(1).expand(-1, batch_size, -1)
        meta_expanded_t = meta.unsqueeze(0).expand(batch_size, -1, -1)
        meta_pairs = torch.cat((meta_expanded, meta_expanded_t), dim=-1)
        
        # Compute temporal similarity
        temporal_expanded = temporal.unsqueeze(1).expand(-1, batch_size)
        temporal_pairs = torch.exp(-torch.abs(temporal_expanded - temporal_expanded.t()) / 30.0)
        
        # Compute edge weights
        meta_sim = self.similarity_net(meta_pairs).squeeze(-1)
        edge_weights = meta_sim * temporal_pairs
        edge_weights = edge_weights * (1 - torch.eye(batch_size, device=device))  # No self-loops
        
        # Create edge indices for the graph
        edge_index = torch.nonzero(edge_weights > 0.1, as_tuple=False).t()
        edge_attr = edge_weights[edge_index[0], edge_index[1]]
        return edge_index, edge_attr

    def forward(self, meta, temporal):
        batch_size = meta.size(0)
        
        # Construct dynamic graph
        edge_index, edge_attr = self.construct_dynamic_graph(meta, temporal)
        
        # Use metadata as node features
        x = meta
        
        # Apply GAT layers
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        
        # Global attention
        x = x.unsqueeze(0)  # [1, B, D]
        global_out, _ = self.global_attention(x, x, x)
        global_out = global_out.squeeze(0)  # [B, D]
        
        # Incorporate temporal features
        temporal_features = self.temporal_mlp(temporal.unsqueeze(-1))
        x = x + temporal_features
        
        # Combine with global attention
        x = global_out + x  # Residual connection
        x = self.fc(x)
        return x