import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class PredictionMapDistillation(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PredictionMapDistillation, self).__init__()
    def forward(self, y, teacher_scores, T=4) :
        """
        basic KD loss function based on "Distilling the Knowledge in a Neural Network"
        https://arxiv.org/abs/1503.02531
        :param y: student score map
        :param teacher_scores: teacher score map
        :param T:  for softmax
        :return: loss value
        """
        p = F.log_softmax(y / T, dim=1)
        q = F.softmax(teacher_scores / T, dim=1)

        p = p.view(-1, 2)
        q = q.view(-1, 2)

        l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
        return l_kl