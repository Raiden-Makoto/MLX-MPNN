import mlx.core as mx # type: ignore
import mlx.nn as nn # type: ignore
from mlx_graphs.nn import global_mean_pool # type: ignore
from EdgeNetwork import EdgeNetwork

class MPNNMLX(nn.Module):
    def __init__(self, nodedim=12, edgedim=4, messagedim=32, numsteps=4, hiddendim=256):
        super().__init__()
        self.messagedim = messagedim
        self.numsteps = numsteps
        self.nodelin = nn.Linear(nodedim, messagedim)  # input projection
        self.edgenet = EdgeNetwork(edgedim, messagedim)
        
        # Implement GRUCell manually with separate gates
        # Reset gate: r = sigmoid(W_ir @ x + W_hr @ h)
        # Update gate: z = sigmoid(W_iz @ x + W_hz @ h)
        # New gate: n = tanh(W_in @ x + r * (W_hn @ h))
        # Output: h' = (1 - z) * n + z * h
        self.W_ir = nn.Linear(messagedim, messagedim)
        self.W_hr = nn.Linear(messagedim, messagedim)
        self.W_iz = nn.Linear(messagedim, messagedim)
        self.W_hz = nn.Linear(messagedim, messagedim)
        self.W_in = nn.Linear(messagedim, messagedim)
        self.W_hn = nn.Linear(messagedim, messagedim)
        
        self.readout = nn.Sequential(
            nn.Linear(messagedim, hiddendim),
            nn.ReLU(),
            nn.Linear(hiddendim, 1),
        )

    def gru_cell(self, x, h):
        """Manual GRU cell implementation"""
        r = mx.sigmoid(self.W_ir(x) + self.W_hr(h))
        z = mx.sigmoid(self.W_iz(x) + self.W_hz(h))
        n = mx.tanh(self.W_in(x) + r * self.W_hn(h))
        h_new = (1 - z) * n + z * h
        return h_new

    def __call__(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.nodelin(x)
        for _ in range(self.numsteps):
            m = self.edgenet(h, edge_index, edge_attr)
            h = self.gru_cell(m, h)
        hg = global_mean_pool(h, batch)
        return mx.sigmoid(mx.squeeze(self.readout(hg), -1))
