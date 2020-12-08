import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss


def nll_loss(output, target):
    return F.nll_loss(output, target)

CELoss = CrossEntropyLoss()
BCELoss = BCELoss()

def vae_loss(recon_x, mu, logvar, x, lm=1e-4):
    loss_fn = MSELoss()
    loss_recon = loss_fn(recon_x, x)
    kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    return loss_recon + lm * kl_divergence, loss_recon, kl_divergence
