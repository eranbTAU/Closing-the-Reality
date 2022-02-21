import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import functional
import torch
from torch.autograd import Variable

def weighted_mse_loss(input,target):
    #alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
    weights = Variable(torch.Tensor([1, 1.5, 2])).cuda() #1, 1.5, 2
    pct_var = (input-target)**2
    out = pct_var * weights.expand_as(target)
    loss = out.mean()
    return loss

def huber_loss(y, y_pred, sigma=0.1):
    r = (y - y_pred).abs()
    loss = (r[r <= sigma]).pow(2).mean()
    loss += (r[r > sigma]).mean() * sigma - sigma**2/2
    return loss

def logcosh(true, pred):
    loss = torch.log(torch.cosh(pred - true))
    return torch.sum(loss) / len(loss)

def XTanhLoss(y_t, y_prime_t):
    ey_t = y_t - y_prime_t
    return torch.mean(ey_t * torch.tanh(ey_t))

def XSigmoidLoss(y_t, y_prime_t):
    ey_t = y_t - y_prime_t
    return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)

def KLDivLoss(inputs, targets):
    inputs = inputs.view(-1,1)
    targets = targets.view(-1,1)
    a = torch.log_softmax(inputs, dim=1)
    b = torch.softmax(targets, dim=1)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    output = kl_loss(a, b)
    return output

class QuantileLoss(nn.Module):
    def __init__(self):
        super(QuantileLoss, self).__init__()
        self.losses = []

    def forward(self, preds, target, quantiles = (0.2, 0.5)):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        for i, q in enumerate(quantiles):
            errors = target[:,i] - preds[:, i]
            self.losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(self.losses, dim=1), dim=1))
        return loss