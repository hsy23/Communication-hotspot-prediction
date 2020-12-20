import torch.nn as nn
import torch


class LSTMs(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, layers, dropout=0):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

        self.softmax = nn.LogSoftmax(dim=1)

        self.sigmoid = nn.Sigmoid() # 最后结果经过sigmoid处理，然后使用BCE_loss

    def forward(self, x):
        out, hidden = self.lstm(x)

        last_hidden_out = out[:, -1, :].squeeze()

        # return last_hidden_out, self.sigmoid(self.linear(last_hidden_out))
        return last_hidden_out, self.linear(last_hidden_out)
