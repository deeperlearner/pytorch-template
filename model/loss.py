import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


def nll_loss(output, target):
    return F.nll_loss(output, target)

CELoss = CrossEntropyLoss()

lm = 1e-5
def vae_loss(recon_x, mu, logvar, x):
    loss_fn = MSELoss()
    loss_recon = loss_fn(recon_x, x)
    kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    return loss_recon
