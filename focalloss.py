import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def compute_class_weight(labels, num_classes):
    """Make class weights for cross-entropy loss (useful for class imbalance,)
    in accordance with sklearn.utils.class_weight.compute_class_weight"""
    weights = len(labels) / (num_classes * torch.bincount(labels))
    return weights

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim()>2:
            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1,2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1,inputs.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        #print("Size of target = " + str(target.size()))
        #print("Size of input = " + str(inputs.size()))
        #print("alpha = " + str(self.alpha) + ", gamma = " + str(self.gamma))

        logpt = F.log_softmax(inputs)
        #print("logpt 1 = " + str(logpt))
        #print("logpt size = " + str(logpt.size()))
        #print(target)
        logpt = logpt.gather(1,target.clone())
        #print("logpt 2 = " + str(logpt))
        logpt = logpt.view(-1)
        #print("logpt 3 = " + str(logpt))
        pt = Variable(logpt.data.exp())
        #print("logpt 4 = " + str(logpt))

        if self.alpha is not None:
            if self.alpha.type()!=inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0,target.clone().data.view(-1))
            logpt = logpt * Variable(at)

        #print("pt = " + str(pt))
        #print("gamma = " + str(self.gamma))
        #print("logpt 5 = " + str(logpt))

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
