import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.ef_dim = 64

        self.conv1 = nn.Conv2d(2, self.ef_dim, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(self.ef_dim, self.ef_dim * 2, 5, stride=2, padding=2)
        self.fc = nn.Linear(16 * 16 * self.ef_dim * 2, self.latent_dim)

    def forward(self, x):
        x = x.view(-1, 2, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 16 * self.ef_dim * 2)
        x = F.relu(self.fc(x))
        return x

class ConvDecoder(nn.Module):
    def __init__(self, observation_dim):
        super(ConvDecoder, self).__init__()
        self.observation_dim = observation_dim
        self.image_size = 64
        self.s2, self.s4, self.s8, self.s16 = int(self.image_size / 2), int(self.image_size / 4), int(self.image_size / 8), int(self.image_size / 16)
        self.gf_dim = 64
        self.c_dim = 2

        self.fc = nn.Linear(self.observation_dim, 16 * 16 * self.gf_dim)
        self.deconv1 = nn.ConvTranspose2d(self.gf_dim, 32, 5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(32, self.c_dim, 5, stride=2, padding=2)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, self.gf_dim, 16, 16)
        x = F.relu(self.deconv1(x))
        x = F.sigmoid(self.deconv2(x))
        x = x.view(-1, self.observation_dim)
        return x

class ConvAnimeEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvAnimeEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.ef_dim = 64

        self.conv1 = nn.Conv2d(2, self.ef_dim, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(self.ef_dim, self.ef_dim * 2, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(self.ef_dim * 2, self.ef_dim * 4, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(self.ef_dim * 4, self.ef_dim * 8, 5, stride=2, padding=2)
        self.fc = nn.Linear(self.ef_dim * 8 * self.s16 * self.s16, self.latent_dim)

    def forward(self, x):
        x = x.view(-1, 2, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.ef_dim * 8 * self.s16 * self.s16)
        x = F.relu(self.fc(x))
        return x

class ConvAnimeDecoder(nn.Module):
    def __init__(self, observation_dim):
        super(ConvAnimeDecoder, self).__init__()
        self.observation_dim = observation_dim
        self.image_size = 64
        self.s2, self.s4, self.s8, self.s16 = int(self.image_size / 2), int(self.image_size / 4), int(self.image_size / 8), int(self.image_size / 16)
        self.gf_dim = 64
        self.c_dim = 1

        self.fc = nn.Linear(self.observation_dim, self.gf_dim * 4 * self.s8 * self.s8)
        self.deconv1 = nn.ConvTranspose2d(self.gf_dim * 4, self.gf_dim * 2, 5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(self.gf_dim * 2, self.gf_dim / 2, 5, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(self.gf_dim / 2, self.c_dim, 5, stride=2, padding=2)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, self.gf_dim * 4, self.s8, self.s8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.sigmoid(self.deconv3(x))
        x = x.view(-1, self.observation_dim)
        return x