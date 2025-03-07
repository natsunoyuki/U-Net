# https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html
# Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

from torchvision.ops import sigmoid_focal_loss


class FocalLoss():
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def __call__(self, logits, targets):
        loss = sigmoid_focal_loss(
            logits, # These must be logits and not probabilities!
            targets, 
            alpha=self.alpha, 
            gamma=self.gamma, 
            reduction=self.reduction,
        ) 
        return loss
