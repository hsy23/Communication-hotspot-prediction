import torch.nn as nn
import torch


class LSTMs(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, layers, dropout=0):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.lstm(x)

        last_hidden_out = out[:, -1, :].squeeze()

        return last_hidden_out, self.sigmoid(self.linear(last_hidden_out))
