import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):

    def __init__(self, latent_dim, batch_size, encoder, decoder,
                 observation_dim=784,
                 learning_rate=1e-3,
                 optimizer=optim.RMSprop,
                 ):

        super(Autoencoder, self).__init__()

        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder(latent_dim)
        self._decode = decoder(observation_dim)
        self._observation_dim = observation_dim
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._build_graph()

    def _build_graph(self):
        self.encode = self._encode
        self.decode = self._decode

        self.loss_function = nn.MSELoss()
        self.optimizer = self._optimizer(self.parameters(), lr=self._learning_rate)

    def forward(self, x):
        encoded = self.encode(x)
        self.z = encoded

        decoded_ch0 = self.decode(self.z, self._observation_dim / 2)
        decoded_ch1 = self.decode(self.z, self._observation_dim / 2)
        self.sample = torch.cat([decoded_ch0, decoded_ch1], axis=-1)

        loss = self.loss_function(x, self.sample)

        return loss

    def update(self, x):
        self.optimizer.zero_grad()
        loss = self.forward(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def x2z(self, x):

        with torch.no_grad():
            self.z = self.encode(x)

        return self.z.numpy().reshape(-1, self._latent_dim)

    def z2x(self, z):

        with torch.no_grad():
            x = self.decode(z, self._observation_dim)

        return x.numpy()

    def reconstruct(self, x):

        with torch.no_grad():
            sample = self.forward(x)

        return sample.numpy()