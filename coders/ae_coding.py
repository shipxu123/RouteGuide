import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvEncoder, self).__init__()
        self.z_dim = latent_dim
        self.ef_dim = 64

        self.conv1 = nn.Conv2d(2, self.ef_dim, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(self.ef_dim, self.ef_dim*2, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(16*16*self.ef_dim*2, self.z_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, 2, 64, 64)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.activation(self.fc(x))
        return x

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoder, self).__init__()

        self.image_size = 64
        self.s2, self.s4, self.s8, self.s16 = int(self.image_size / 2), int(self.image_size / 4), int(self.image_size / 8), int(self.image_size / 16)
        self.gf_dim = 64
        self.c_dim = 1

        self.fc = nn.Linear(latent_dim, 16*16*self.gf_dim)
        self.deconv1 = nn.ConvTranspose2d(self.gf_dim, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, self.c_dim, kernel_size=4, stride=2, padding=1)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

    def forward(self, z):
        x = self.activation1(self.fc(z))
        x = x.view(-1, self.gf_dim, 16, 16)
        x = self.activation1(self.deconv1(x))
        x = self.activation2(self.deconv2(x))
        x = x.view(-1, self.image_size*self.image_size*self.c_dim)
        return x

class ConvDecoder2ch(nn.Module):
    def __init__(self, observation_dim):
        super(ConvDecoder2ch, self).__init__()

        self.image_size = 64
        self.s2, self.s4, self.s8, self.s16 = int(self.image_size / 2), int(self.image_size / 4), int(self.image_size / 8), int(self.image_size / 16)
        self.gf_dim = 64
        self.c_dim = 1

        self.fc = nn.Linear(observation_dim, 16*16*self.gf_dim)
        self.deconv1 = nn.ConvTranspose2d(self.gf_dim, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, self.c_dim, kernel_size=4, stride=2, padding=1)
        self.deconv1b = nn.ConvTranspose2d(self.gf_dim, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2b = nn.ConvTranspose2d(32, self.c_dim, kernel_size=4, stride=2, padding=1)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

    def forward(self, z):
        x = self.activation1(self.fc(z))
        x = x.view(-1, self.gf_dim, 16, 16)
        x1 = self.activation1(self.deconv1(x))
        x1 = self.activation2(self.deconv2(x1))
        x1 = x1.view(-1, self.image_size*self.image_size*self.c_dim)
        x2 = self.activation1(self.deconv1b(x))
        x2 = self.activation2(self.deconv2b(x2))
        x2 = x2.view(-1, self.image_size*self.image_size*self.c_dim)
        return torch.cat((x1, x2), 0).shape