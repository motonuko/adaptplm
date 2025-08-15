import torch
from torch import nn


class FeedForwardNN2(nn.Module):
    def __init__(self, prot_input_dim, mol_input_dim, hidden_dim, n_hidden_layers, dropout_rate):
        super(FeedForwardNN2, self).__init__()
        assert n_hidden_layers >= 1
        self.mol_fc = nn.Sequential(
            nn.Linear(mol_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.prot_fc = nn.Sequential(
            nn.Linear(prot_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        layers = [nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_prot, x_mol):
        prot_out = self.prot_fc(x_prot)
        mol_out = self.mol_fc(x_mol)
        combined = torch.cat((prot_out, mol_out), dim=1)
        logit = self.layers(combined)
        output = self.sigmoid(logit)
        return {'sigmoid_output': output, 'logit': logit}
