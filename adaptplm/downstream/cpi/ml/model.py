# from torch import nn
#
#
# class FeedForwardNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, n_hidden_layers, dropout_rate):
#         super(FeedForwardNN, self).__init__()
#         layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
#         for _ in range(n_hidden_layers):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout_rate))
#         layers.append(nn.Linear(hidden_dim, 1))
#         layers.append(nn.Sigmoid())
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.model(x)
#         return out
