import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def multiscale_supervision(gt_occ, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''

    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]]).to(gt_occ.device).type(torch.float) 
    for i in range(gt.shape[0]):
        coords = gt_occ[i][:, :3].type(torch.long) // ratio
        gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_occ[i][:, 3]
    
    return gt

def geo_scal_loss(pred, ssc_target, semantic=True):

    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        # empty_probs = pred[:, 0, :, :, :]
        empty_probs = pred[:, 0]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )

# 0.95  5  3
# 0.9   5  3
# 0.95  3  3

class AdaptiveLoss(nn.Module):

    def __init__(self, calsses_num=17, beta=0.95, alpha=5, weight_per_class=3):
        super(AdaptiveLoss, self).__init__()
        self.calsses_num = calsses_num
        F1_list = torch.zeros(calsses_num)
        self.register_buffer('F1_list', F1_list)
        self.beta = beta
        self.alpha = alpha
        self.weight_per_class = weight_per_class

    def forward(self, pred, ssc_target):
        return self.sem_scal_loss(pred, ssc_target)

    def sem_scal_loss(self, pred, ssc_target):
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)
        loss = 0
        count = 0
        mask = ssc_target != 255
        n_classes = self.calsses_num

        cur_F1_list = torch.zeros(n_classes).cuda()
        loss_list = torch.zeros(n_classes).cuda()

        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i]

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0

            try:
                if torch.sum(completion_target) > 0:
                    count += 1.0
                    nominator = torch.sum(p * completion_target)
                    loss_class = 0
                    precision = 0
                    recall = 0
                    if torch.sum(p) > 0:
                        precision = nominator / (torch.sum(p))
                        loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
                        loss_list[i] += loss_precision
                    if torch.sum(completion_target) > 0:
                        recall = nominator / (torch.sum(completion_target))
                        loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                        loss_list[i] += loss_recall

                    F1 = 2 * precision * recall / (precision + recall)
                    cur_F1_list[i] = F1

                    if torch.sum(1 - completion_target) > 0:
                        specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target))
                        loss_specificity = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
                        loss_list[i] += loss_specificity

            except:
                print("!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!")

        weight_pool = self.weight_per_class * count
        self.F1_list.data = self.beta*self.F1_list.data + (1-self.beta)*cur_F1_list
        cur_F1_list = self.F1_list[loss_list != 0]
        loss_list = loss_list[loss_list != 0]
        cur_F1_list_softmax = F.softmax(self.alpha * (1 - cur_F1_list), dim=0)
        loss_list = loss_list * (1 + weight_pool * cur_F1_list_softmax)
        loss = torch.sum(loss_list)
        return loss / (count * (1 + self.weight_per_class))


class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='none', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count