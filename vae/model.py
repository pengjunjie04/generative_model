import torch.nn as nn
import torch
class ConvVAE(nn.Module):
    def __init__(self,latent_dim=128,ch=64,image_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = ch
        self.image_size = image_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3,ch,4,2,1),
            nn.ReLU(True),
            nn.Conv2d(ch,ch*2,4,2,1),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True)           
        )

        with torch.no_grad():
            dummy = torch.zeros(1,3,image_size,image_size)
            feat_dim = self.encoder(dummy).view(1,-1).size(1) # feat_dim = C*H*W

        self.fc_mu = nn.Linear(feat_dim,latent_dim)
        self.fc_logvar = nn.Linear(feat_dim,latent_dim)
        self.fc_dec = nn.Linear(latent_dim,feat_dim)

        self._feat_shape = self.encoder(dummy).shape[1:] # (C,H,W)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),
            nn.Sigmoid() 
        )

    def encode(self,x):

        h = self.encoder(x)
        h = h.view(h.size(0),-1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu,logvar

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std*eps # eps就是标准高斯噪声的采样结果

    def decode(self,z):
        h = self.fc_dec(z)
        h = h.view(h.size(0),*self._feat_shape)

        x_recon = self.decoder(h)
        return x_recon

    def forward(self,x):
        mu,logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        x_recon = self.decode(z)
        return x_recon,mu,logvar
