"""
lstm_gru.py — Baseline Models for Table 1 Comparison

Standard LSTM and GRU baselines that treat all sensor channels as an
independent flattened input vector — NO graph structure, NO physics.

These represent the "current industry standard" your paper compares against.
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Standard 2-layer LSTM baseline.

    Input:  [B, T, N, F] — flattened to [B, T, N*F] before LSTM
    Output: [B, N]       — RUL prediction per node
    """
    def __init__(self, config: dict):
        super().__init__()
        n_nodes    = config['graph']['n_nodes']
        n_features = config['preprocessing']['n_features']
        hidden_dim = config['model']['agcrn']['hidden_dim']

        self.n_nodes = n_nodes
        input_dim = n_nodes * n_features

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.readout = nn.Linear(hidden_dim, n_nodes)

    def forward(self, X_seq: torch.Tensor) -> dict:
        B, T, N, F = X_seq.shape
        X_flat = X_seq.reshape(B, T, N * F)      # flatten sensors — no graph structure
        _, (h_n, _) = self.lstm(X_flat)
        h_last = h_n[-1]                          # [B, hidden_dim] — last layer's final state
        rul = torch.sigmoid(self.readout(h_last))  # [B, N]
        return {'rul': rul}


class GRUBaseline(nn.Module):
    """
    Standard 2-layer GRU baseline — same structure as LSTM but simpler cell.
    """
    def __init__(self, config: dict):
        super().__init__()
        n_nodes    = config['graph']['n_nodes']
        n_features = config['preprocessing']['n_features']
        hidden_dim = config['model']['agcrn']['hidden_dim']

        self.n_nodes = n_nodes
        input_dim = n_nodes * n_features

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.readout = nn.Linear(hidden_dim, n_nodes)

    def forward(self, X_seq: torch.Tensor) -> dict:
        B, T, N, F = X_seq.shape
        X_flat = X_seq.reshape(B, T, N * F)
        _, h_n = self.gru(X_flat)
        h_last = h_n[-1]
        rul = torch.sigmoid(self.readout(h_last))
        return {'rul': rul}
