import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss


def weighted_bce_loss(output, target, weight: torch.Tensor=None):
    weighted_bce_loss = BCELoss(weight=weight[target.long()])
    return weighted_bce_loss(output, target)


# ref: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
def binary_focal_loss(output: torch.Tensor,
                      target: torch.Tensor,
                      alpha: float = 0.5,
                      gamma: float = 2.0,
                      reduction: str = 'sum',
                      eps: float = 1e-8) -> torch.Tensor:
    p_t = output
    loss_tmp = -alpha * torch.pow(1 - p_t, gamma) * target * torch.log(p_t + eps) \
               - (1 - alpha) * torch.pow(p_t, gamma) * (1 - target) * torch.log(1 - p_t + eps)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)

    return loss


# model output with log_softmax
def nll_loss(output, target):
    return F.nll_loss(output, target)


def vae_loss(recon_x, mu, logvar, x, lm=1e-4):
    mse_loss = MSELoss()
    loss_recon = mse_loss(recon_x, x)
    kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    return loss_recon + lm * kl_divergence, loss_recon, kl_divergence
