import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import functional
import torch

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        return lossvalue

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]
    
class KLDivLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(KLDivLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        x = F.log_softmax(inputs)
        a = torch.log_softmax(inputs, dim=1)
        b = torch.softmax(targets, dim=1)
        kl_loss = nn.KLDivLoss(reduction = 'batchmean')
        output = kl_loss(a, b)
        return output

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class VarianceLayer(nn.Module):
    def __init__(self, patch_size=5, channels=1):
        self.patch_size = patch_size
        super(VarianceLayer, self).__init__()
        mean_mask = np.ones((channels, channels, patch_size, patch_size)) / (patch_size * patch_size)
        self.mean_mask = nn.Parameter(data=torch.FloatTensor(mean_mask), requires_grad=False)
        mask = np.zeros((channels, channels, patch_size, patch_size))
        mask[:, :, patch_size // 2, patch_size // 2] = 1.
        self.ones_mask = nn.Parameter(data=torch.FloatTensor(mask), requires_grad=False)

    def forward(self, x):
        Ex_E = F.conv2d(x, self.ones_mask) - F.conv2d(x, self.mean_mask)
        return F.conv2d((Ex_E) ** 2, self.mean_mask)
