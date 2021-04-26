import torch
from torch import nn

from tcn_embed import TCN

class EmbedNet(nn.Module):
    def __init__(self, input_channels, n_channels, kernel_size, dropout, lstm_n_hidden, lstm_n_layers, lstm_bidirectional, n_classes=4):
        super(EmbedNet, self).__init__()
        # 1 TCN for all questions
        self.tcn = TCN(input_channels, n_channels, kernel_size, dropout)
        self.lstm = nn.GRU(n_channels[-1], lstm_n_hidden, lstm_n_layers, True, False, dropout if lstm_n_layers > 0 else 0, lstm_bidirectional)
#         self.lstm = nn.LSTM(n_channels[-1], lstm_n_hidden, dropout, lstm_bidirectional)
        self.fc = nn.Linear((2 if lstm_bidirectional else 1) * lstm_n_hidden, n_classes)
        self.regressor = n_classes == 1

    def forward(self, inputs):
        embeds = torch.stack([self.tcn(inpt) for inpt in inputs])
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[-1]
        out = self.fc(lstm_out)
        return out

    def train_forward(self, data, criterion, device=None):
        inputs, phq_buckets = data
        inputs = [inpt.float() for inpt in inputs]
        phq_buckets = phq_buckets.float()
        if device:
            inputs = [inpt.to(device) for inpt in inputs]
            phq_buckets = phq_buckets.to(device)
        phq_preds = self.forward(inputs)
        if self.regressor:
            phq_buckets = phq_buckets.view(-1, 1)
        loss = criterion(phq_preds, phq_buckets.long())
        return loss


