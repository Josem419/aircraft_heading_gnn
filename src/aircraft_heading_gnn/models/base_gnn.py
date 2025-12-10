"""
Base GNN architectures for heading prediction.
Implements GCN and GAT variants for spatial reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from typing import Optional


class BaseGNN(nn.Module):
    """
    Base Graph Neural Network for heading prediction.
    Uses Graph Convolutional layers to aggregate spatial information.
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 72,
        dropout: float = 0.2,
        use_edge_features: bool = True,
    ):
        """
        Args:
            num_node_features: Input node feature dimension
            num_edge_features: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            num_classes: Number of heading classes (72 for 5-degree bins)
            dropout: Dropout rate
            use_edge_features: Whether to use edge features
        """
        super().__init__()

        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_edge_features = use_edge_features

        # Input projection
        self.node_encoder = nn.Linear(num_node_features, hidden_dim)

        if use_edge_features:
            self.edge_encoder = nn.Linear(num_edge_features, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
            batch: Batch assignment for nodes

        Returns:
            Logits for each node [num_nodes, num_classes]
        """
        # Encode node features
        h = self.node_encoder(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Apply GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Residual connection
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        # Classify each node
        out = self.classifier(h)

        return out


class GATHeadingPredictor(nn.Module):
    """
    Graph Attention Network for heading prediction.
    Uses attention to weight contributions from nearby aircraft.
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 72,
        num_heads: int = 4,
        dropout: float = 0.2,
        edge_dim: Optional[int] = None,
    ):
        """
        Args:
            num_node_features: Input node feature dimension
            num_edge_features: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GAT layers
            num_classes: Number of heading classes
            num_heads: Number of attention heads
            dropout: Dropout rate
            edge_dim: Edge feature dimension for GAT (uses num_edge_features if None)
        """
        super().__init__()

        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout

        if edge_dim is None:
            edge_dim = num_edge_features

        # Input projection
        self.node_encoder = nn.Linear(num_node_features, hidden_dim)

        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: single head
                self.convs.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim,
                        heads=1,
                        dropout=dropout,
                        edge_dim=edge_dim if i == 0 else None,
                    )
                )
            else:
                # Multi-head attention
                self.convs.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim if i == 0 else None,
                        concat=True,
                    )
                )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output head with auxiliary task
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Auxiliary head for turn detection (left/straight/right)
        self.turn_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 3),  # left, straight, right
        )

    def forward(
        self, x, edge_index, edge_attr=None, batch=None, return_attention=False, return_dict=False
    ):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
            batch: Batch assignment for nodes
            return_attention: Whether to return attention weights
            return_dict: If True, returns dict with all tasks. If False, returns only heading tensor.

        Returns:
            Tensor [num_nodes, num_classes] if return_dict=False
            Dictionary with 'heading' logits and optionally 'turn_direction' if return_dict=True
        """
        # Encode node features
        h = self.node_encoder(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        attention_weights = []

        # Apply GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if return_attention and i == 0:
                h_new, attn = conv(
                    h, edge_index, edge_attr, return_attention_weights=True
                )
                attention_weights.append(attn)
            else:
                if i == 0:
                    h_new = conv(h, edge_index, edge_attr)
                else:
                    h_new = conv(h, edge_index)

            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Residual connection
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        # Main task: heading classification
        heading_out = self.classifier(h)

        # Auxiliary task: turn direction
        turn_out = self.turn_detector(h)

        # Return format based on return_dict parameter
        if return_dict:
            outputs = {"heading": heading_out, "turn_direction": turn_out}
            if return_attention:
                outputs["attention_weights"] = attention_weights
            return outputs
        else:
            # Simple tensor output for compatibility with training scripts
            return heading_out
