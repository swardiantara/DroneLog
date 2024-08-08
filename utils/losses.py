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
    

class SeverityFocal(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(SeverityFocal, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        y_pred_probs = torch.softmax(inputs, axis=1)

        # Compute weights based on severity difference with higher penalty for false positives
        true_severity = targets.float()
        predicted_severity = y_pred_probs.argmax(dim=1).float()
        miss_penalty = torch.where(true_severity > predicted_severity,
                              2.5 * (1 + torch.abs(true_severity - predicted_severity)),
                              1 + torch.abs(true_severity - predicted_severity))
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss) * miss_penalty
        return loss.mean()
    

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
        focal_loss = (self.alpha[class_idx] * (1 - p_t) ** self.gamma * bce_loss)
        
        return focal_loss.mean()
    

class SeverityCE(nn.Module):
    def __init__(self, reduction='mean', class_weights=None):
        super(SeverityCE, self).__init__()
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, y_pred_logits, y_true):
        # Apply softmax to obtain class probabilities
        y_pred_probs = torch.softmax(y_pred_logits, axis=1)

        # Compute weights based on severity difference with higher penalty for false positives
        true_severity = y_true.float()
        predicted_severity = y_pred_probs.argmax(dim=1).float()
        weights = torch.where(true_severity > predicted_severity,
                              2.5 * (1 + torch.abs(true_severity - predicted_severity)),
                              1 + torch.abs(true_severity - predicted_severity))
        
        # Compute weighted cross-entropy loss
        loss = F.cross_entropy(y_pred_logits, y_true, weight=self.class_weights) * weights

        # Apply reduction
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        # If reduction is None, return the raw loss tensor
        elif self.reduction is None:
            pass
        else:
            raise ValueError("Invalid reduction option. Please use 'mean', 'sum', or None.")

        return loss
    
    
# Define the Ordinal Cross-Entropy Loss
class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, weight=None):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, logits, targets):
        logits = logits.view(-1, self.num_classes)
        targets = targets.view(-1)

        # Compute the ordinal cross-entropy loss
        ordinal_loss = torch.tensor(0.0, requires_grad=True)
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                ordinal_loss += torch.sum(torch.clamp(logits[:, i] - logits[:, j], min=0) * (targets == j))

        return ordinal_loss