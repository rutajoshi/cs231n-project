import torch
from torch import nn

class AE(nn.Module):
    def __init__(self, input_channels, shape_x, shape_y, n_classes=4):
        super(AE, self).__init__()
        # Encoder layers
        self.enc_conv1 = nn.ReLU(nn.Conv2D(input_channels, 1, 5, padding=1))
        self.enc_pool1 = nn.MaxPool2d(2, padding=0)
        self.enc_conv2 = nn.ReLU(nn.Conv2D(1, 1, 5, padding=0))
        self.enc_pool2 = nn.MaxPool2d(2, padding=0)
        # Decoder layers
        self.dec_conv1 = nn.Conv2D(1, 1, 5, padding=0)
        self.dec_unpool1 = 

    def forward(self, inputs):
        # Input must be a 48x48 image, 1 channel
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        out = self.pool2(x)
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


