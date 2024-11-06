import re
import math
import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Bernoulli
import coders.ae_coding as aec

class VAE(nn.Module):
    def __init__(self, latent_dim, batch_size, encoder, decoder,
                 observation_dim=784,
                 learning_rate=3e-4,
                 optimizer=torch.optim.RMSprop,
                 observation_distribution="Bernoulli", # or Gaussian
                 observation_std=0.01,
                 image_ch_dim=2,
                 decode_2ch=aec.ConvDecoder2ch):
        super(VAE, self).__init__()

        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._observation_dim = observation_dim
        self._learning_rate = learning_rate
        self._observation_distribution = observation_distribution
        self._observation_std = observation_std
        self._image_ch_dim = image_ch_dim

        self.encoder = encoder(latent_dim * 2)
        self.decoder = decoder(latent_dim)
        self.decoder_2ch = decode_2ch(latent_dim)

        self._optimizer = optimizer
        self.optimizer = self._optimizer(self.parameters(), lr=self._learning_rate)

    def _kl_diagnormal_stdnormal(self, mu, log_var):
        var = torch.exp(log_var)
        kl = 0.5 * torch.sum(torch.square(mu) + var - 1. - log_var)
        return kl

    def _gaussian_log_likelihood(self, targets, mean, std):
        se = 0.5 * torch.sum((targets - mean) * (targets - mean)) / (2 * std * std) + math.log(std)
        return se

    def _bernoulli_log_likelihood(self, targets, outputs, eps=1e-8):
        log_like = -torch.sum(targets * torch.log(outputs + eps)
                                  + (1. - targets) * torch.log((1. - outputs) + eps))
        return log_like

    def forward(self, x):
        encoded = self.encoder(x)
        self.mean = encoded[:, :self._latent_dim]
        logvar = encoded[:, self._latent_dim:]
        stddev = torch.sqrt(torch.exp(logvar))
        epsilon = torch.randn([self._batch_size, self._latent_dim]).cuda()
        self.z = self.mean + stddev * epsilon

        if self._image_ch_dim == 2:
            decoded_0 = self.decoder(self.z)
            obs_mean_0 = decoded_0
            if self._observation_distribution == 'Gaussian':
                obs_epsilon = torch.randn([self._batch_size, self._observation_dim // 2]).cuda()
                sample_0 = obs_mean_0 + self._observation_std * obs_epsilon
            else:
                sample_0 = Bernoulli(probs=obs_mean_0).sample()

            decoded_1 = self.decoder(self.z)
            obs_mean_1 = decoded_1
            if self._observation_distribution == 'Gaussian':
                obs_epsilon = torch.randn([self._batch_size, self._observation_dim // 2]).cuda()
                sample_1 = obs_mean_1 + self._observation_std * obs_epsilon
            else:
                sample_1 = Bernoulli(probs=obs_mean_1).sample()

            self.obs_mean = torch.cat([obs_mean_0, obs_mean_1], dim=-1)
            self.sample = torch.cat([sample_0, sample_1], dim=-1)

        elif self._image_ch_dim == 3:
            decoded_0 = self.decoder(self.z)
            obs_mean_0 = decoded_0
            if self._observation_distribution == 'Gaussian':
                obs_epsilon = torch.randn([self._batch_size, self._observation_dim // 3]).cuda()
                sample_0 = obs_mean_0 + self._observation_std * obs_epsilon
            else:
                sample_0 = Bernoulli(probs=obs_mean_0).sample()

            decoded_1 = self.decoder_2ch(self.z)
            obs_mean_1 = decoded_1
            if self._observation_distribution == 'Gaussian':
                obs_epsilon = torch.randn([self._batch_size, self._observation_dim * 2 // 3]).cuda()
                sample_1 = obs_mean_1 + self._observation_std * obs_epsilon
            else:
                sample_1 = Bernoulli(probs=obs_mean_1).sample()
            
            self.obs_mean = torch.cat([obs_mean_0, obs_mean_1], dim=-1)
            self.sample = torch.cat([sample_0, sample_1], dim=-1)
            
        kl = self._kl_diagnormal_stdnormal(self.mean, logvar)
        if self._observation_distribution == 'Gaussian':
            obj = self._gaussian_log_likelihood(x, self.obs_mean, self._observation_std)
        else:
            obj = self._bernoulli_log_likelihood(x, self.obs_mean)
        self._loss = (kl + obj) / self._batch_size
        
        return self.obs_mean, self.sample, self._loss

    def update(self, x):
        self.optimizer.zero_grad()
        _, _, loss = self.forward(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def x2z(self, x):
        mean = self.encoder(x)[:, :self._latent_dim]
        return mean.detach().cpu().numpy().reshape(-1, self._latent_dim)

    def z2x(self, z):
        self.z.data = z
        x = self.decoder(self.z).detach().cpu().numpy()
        return x

    def reconstruct(self, x):
        # sample = self.decoder(self.encoder(x)[:,:self._latent_dim])
        _, sample, _ = self.forward(x)
        return sample.detach().cpu().numpy()

    def save_encoder(self, path, prefix="vae/encoder"):
        var_dict = {}
        for name, param in self.state_dict().items():
            if "encoder" in name:
                var_dict[name.replace("encoder.", "")] = param
        torch.save(var_dict, path)

    def save_decoder(self, path, prefix="vae/generator"):
        var_dict = {}
        for name, param in self.state_dict().items():
            if "decoder" in name:
                var_dict[name.replace("decoder.", "")] = param
        torch.save(var_dict, path)


class VAE_GEN(nn.Module):
    def __init__(self, latent_dim, batch_size, encoder, decoder,
                 observation_dim=784,
                 learning_rate=1e-4,
                 optimizer=torch.optim.RMSprop,
                 observation_distribution="Bernoulli", # or Gaussian
                 observation_std=0.01):
        super(VAE_GEN, self).__init__()

        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._observation_dim = observation_dim
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._observation_distribution = observation_distribution
        self._observation_std = observation_std

        self.encoder = encoder(latent_dim * 2)
        self.decoder = decoder(latent_dim)

        self.obs_mean = None
        self.sample = None

        self.optimizer_en = self._optimizer(self.encoder.parameters(), lr=self._learning_rate)
        self.optimizer_de = self._optimizer(self.decoder.parameters(), lr=self._learning_rate)
        self.optimizer_all = self._optimizer(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self._learning_rate/10)


    def forward(self, x):
        encoded = self.encoder(x)
        self.mean = encoded[:, :self._latent_dim]
        self.logvar = encoded[:, self._latent_dim:]
        stddev = torch.sqrt(torch.exp(self.logvar))
        epsilon = torch.randn([self._batch_size, self._latent_dim]).cuda() #Re-parameterization
        z = self.mean + stddev * epsilon

        self.obs_mean = self.decoder(z)
        if self._observation_distribution == 'Gaussian':
            obs_epsilon = torch.randn([self._batch_size, self._observation_dim // 2]).cuda()
            self.sample = self.obs_mean + self._observation_std * obs_epsilon
        else:
            self.sample = torch.bernoulli(self.obs_mean)

        return self.sample

    def _kl_diagnormal_stdnormal(self, mu, log_var):
        var = torch.exp(log_var)
        kl = 0.5 * torch.sum(torch.square(mu) + var - 1. - log_var)
        return kl

    def _gaussian_log_likelihood(self, targets, mean, std):
        se = 0.5 * torch.sum((targets - mean) * (targets - mean)) / (2 * std * std) + math.log(std)
        return se

    def _bernoulli_log_likelihood(self, targets, outputs, eps=1e-8):
        log_like = -torch.sum(targets * torch.log(outputs + eps)
                                  + (1. - targets) * torch.log((1. - outputs) + eps))
        return log_like

    def update_phrase1(self, x, y):
        self.optimizer_de.zero_grad()
        output = self.forward(x)
        loss = torch.sum(torch.square(y - output)) / self._batch_size
        loss.backward()
        self.optimizer_de.step()
        return loss.item()

    def update_phrase2(self, x, y):
        self.optimizer_all.zero_grad()
        output = self.forward(x)
        kl = self._kl_diagnormal_stdnormal(self.mean, self.logvar)
        if self._observation_distribution == 'Gaussian':
            loss = self._gaussian_log_likelihood(y, output, self._observation_std)
        else:
            loss = self._bernoulli_log_likelihood(y, output)
        loss_phrase2 = (kl + loss) / self._batch_size
        loss_phrase2.backward()
        self.optimizer_all.step()
        return loss_phrase2.item()

    def x2z(self, x):
        encoded = self.encoder(x)
        mean = encoded[:, :self._latent_dim]
        return np.asarray(mean.detach().cpu().numpy()).reshape(-1, self._latent_dim)

    def z2x(self, z):
        x = self.decoder(z)
        return np.asarray(x.detach().cpu().numpy())

    def sample(self, z):
        x = self.decoder(z)
        return np.asarray(x.detach().cpu().numpy())

    def generate(self, x):
        sample = self.forward(x)
        return np.asarray(sample.detach().cpu().numpy())

    def load_pretrained_encoder(self, path):
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict)

    def load_pretrained_decoder(self, path):
        state_dict = torch.load(path)
        self.decoder.load_state_dict(state_dict)
      
    def save_encoder(self, path, prefix="in/encoder"):
        torch.save(self.encoder.state_dict(), path)

    def save_decoder(self, path, prefix="in/generator"):
        torch.save(self.decoder.state_dict(), path)