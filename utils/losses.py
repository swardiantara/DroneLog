import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


def inverse_freq(y_train):
    class_counts = np.bincount(y_train)
    num_classes = len(class_counts)
    total_samples = len(y_train)

    class_weights = []
    for count in class_counts:
        weight = 1 / (count / total_samples)
        class_weights.append(weight)
        
    return class_weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
    

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Compute the BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        print(f"targets: {targets}")
        if targets[0] == 1:
            class_idx = 0
        else:
            if targets[2] == 1:
                class_idx = 2
            elif targets[1] == 1:
                class_idx = 1
            else:
                class_idx = 0
        # Compute the sigmoid and modulate it with the alpha and gamma factors
        p_t = torch.exp(-bce_loss)
        focal_loss = (self.alpha[class_idx] * (1 - p_t) ** self.gamma) * bce_loss
        
        return focal_loss.mean()