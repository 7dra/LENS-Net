

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean', eps=1e-10):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        eps = self.eps
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt + eps) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + eps)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'False':
            loss = loss
        return loss


def Fbeta_Measure(pred, target):
    b = pred.shape[0]
    precision_mean = 0.0
    recall_mean = 0.0
    Fbeta_mean = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        precision = torch.sum(target[i, :, :, :] * pred[i, :, :, :]) / (torch.sum(pred[i, :, :, :]) + 1e-10)
        recall = torch.sum(target[i, :, :, :] * pred[i, :, :, :]) / (torch.sum(target[i, :, :, :]) + 1e-10)

        # precision = torch.sum(torch.logical_and(pred, target).float()) / (torch.sum(pred[i, :, :, :]) + 1e-10)
        # recall = torch.sum(torch.logical_and(pred, target).float()) / (torch.sum(target[i, :, :, :]) + 1e-10)

        Fbeta = (1.3 * precision * recall) / ((0.3 * precision + recall) + 1e-10)

        precision_mean += precision
        recall_mean += recall
        Fbeta_mean += Fbeta

    return precision_mean / b, recall_mean / b, Fbeta_mean / b


def IOU(pred, gt, extra=False, eps=1e-8):
    inter = (gt * pred).sum(dim=(1, 2, 3))
    union = (gt + pred).sum(dim=(1, 2, 3)) - inter
    iou_loss = 1 - inter / (union + eps)

    # background = (gt == 0).float()

    # background_inter = (background * pred).sum(dim=(1, 2, 3))
    # background_union = background.sum(dim=(1, 2, 3))

    # background_loss = background_inter / (background_union + eps)

    # loss = iou_loss + background_loss

    loss = iou_loss.mean()
    if extra:
        precision = inter / (pred.sum(dim=(1, 2, 3)) + 1e-10)
        recall = inter / (gt.sum(dim=(1, 2, 3)) + 1e-10)
        Fbeta = (1.3 * precision * recall) / ((0.3 * precision + recall) + 1e-10)
        return loss, precision.mean(), recall.mean(), Fbeta.mean()
    return loss


def MAE_Measure(pred, target):
    mae = F.l1_loss(pred, target, reduction='mean')
    return mae


def Structure_Measure(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = torch.tensor(1.0)
    else:
        Q = torch.tensor(0.0)
    return Q
