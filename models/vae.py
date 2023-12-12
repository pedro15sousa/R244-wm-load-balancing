"""
Variational encoder model, used as a visual model
for agent's world view.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, state_size, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.state_size = state_size

        self.fc1 = nn.Linear(latent_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, state_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        reconstruction = torch.sigmoid(self.fc3(x))
        return reconstruction
    
class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, state_size, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.state_size = state_size

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_size)
        self.fc_logsigma = nn.Linear(128, latent_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print("encoder x")
        # print(x)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, state_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(state_size, latent_size)
        self.decoder = Decoder(state_size, latent_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma