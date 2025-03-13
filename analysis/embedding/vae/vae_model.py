import torch
import torch.nn as nn

class DeepVAE(nn.Module):
    def __init__(self, latent_dim, feature_dim, hidden_dim):
        super(DeepVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(feature_dim, hidden_dim, latent_dim)
        self.decoder = self.build_decoder(latent_dim, hidden_dim, feature_dim)

    def build_encoder(self, feature_dim, hidden_dim, latent_dim):
        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, latent_dim * 2)  # Outputting both mean and logvar
        )

    def build_decoder(self, latent_dim, hidden_dim, feature_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def encode(self, x):
        encoded = self.encoder(x)
        mean_logvar = encoded.view(-1, 2, self.latent_dim)
        mean, logvar = mean_logvar[:, 0, :], mean_logvar[:, 1, :]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar

class VanillaVAE(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = self.build_decoder(latent_dim, hidden_dim, input_dim)

    def build_encoder(self, input_dim, hidden_dim, latent_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2)  # Output mean and logvar
        )

    def build_decoder(self, latent_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def encode(self, x):
        encoded = self.encoder(x)
        mean_logvar = encoded.view(-1, 2, self.latent_dim)
        mean, logvar = mean_logvar[:, 0, :], mean_logvar[:, 1, :]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar
