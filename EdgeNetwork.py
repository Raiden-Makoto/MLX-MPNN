import mlx.core as mx # type: ignore
import mlx.nn as nn # type: ignore
from mlx_graphs.nn.message_passing import MessagePassing # type: ignore

class EdgeNetwork(MessagePassing):
    def __init__(self, edge_dim, message_dim):
        super().__init__(aggr='add')
        self.fc = nn.Linear(edge_dim, message_dim * message_dim)
        self.message_dim = message_dim

    def __call__(self, x, edge_index, edge_attr):
        weight = self.fc(edge_attr)
        weight = weight.reshape(-1, self.message_dim, self.message_dim)
        return self.propagate(edge_index, x, message_kwargs={'weight': weight})

    def message(self, src_features, dst_features, **kwargs):
        # src_features shape: [num_edges, message_dim]
        # weight shape: [num_edges, message_dim, message_dim]
        # We need to expand src_features to [num_edges, message_dim, 1] for matrix multiplication
        weight = kwargs['weight']
        src_expanded = mx.expand_dims(src_features, -1)
        # Result shape: [num_edges, message_dim, 1]
        result = mx.matmul(weight, src_expanded)
        # Squeeze to [num_edges, message_dim]
        return mx.squeeze(result, -1)