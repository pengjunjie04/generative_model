import torch.nn.functional as F
import torch

def vae_loss(recon_x,x,mu,logvar):
    recon_loss = F.mse_loss(recon_x,x,reduction="sum")

    kl_divergence = torch.sum(1+logvar - mu**2 - logvar.exp())

    return recon_loss + kl_divergence,recon_loss,kl_divergence

