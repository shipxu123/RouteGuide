import re
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

class MMD_VAE(nn.Module):
    
    def __init__(self, latent_dim, batch_size, encoder, decoder, observation_dim=784, learning_rate=1e-4, 
                 optimizer=torch.optim.RMSprop, observation_distribution="Bernoulli", observation_std=0.01):
        super(MMD_VAE, self).__init__()
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
        self._observation_dim = observation_dim
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._build_graph()

    def _build_graph(self):
        self.x = torch.nn.Parameter(torch.randn((self._observation_dim)))
        self.encoder = self._encode(self.x, self._latent_dim)
        self.z = self.encoder
        self.decoder0 = self._decode(self.z, int(self._observation_dim / 2))
        self.decoder1 = self._decode(self.z, int(self._observation_dim / 2))

        self.sample = torch.cat([self.decoder0, self.decoder1], dim=-1)

        self.loss_mmd = self._compute_mmd(torch.randn((self.z.size()[0], self._latent_dim)), self.z)
        self.loss_nll = torch.sum(nn.MSELoss()(self.x, self.sample))

        self._loss = self.loss_mmd + self.loss_nll
        self.optimizer = self._optimizer(learning_rate=self._learning_rate)

    def _compute_mmd(self, x, y):
        x_kernel = self._compute_kernel(x, x)
        y_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def _compute_kernel(self, x, y):
        x_size = x.size()[0]
        y_size = y.size()[0]
        dim = x.size()[1]
        tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean(torch.square(tiled_x - tiled_y), dim=2) / dim)

    def update(self, batch):
        self.optimizer.zero_grad()
        self.x = nn.Parameter(torch.FloatTensor(batch))
        self.encoder = self._encode(self.x, self._latent_dim)
        self.z = self.encoder
        self.decoder0 = self._decode(self.z, int(self._observation_dim / 2))
        self.decoder1 = self._decode(self.z, int(self._observation_dim / 2))

        self.sample = torch.cat([self.decoder0, self.decoder1], dim=-1)

        self.loss_mmd = self._compute_mmd(torch.randn((self.z.size()[0], self._latent_dim)), self.z)
        self.loss_nll = torch.sum(nn.MSELoss()(self.x, self.sample))

        self._loss = self.loss_mmd + self.loss_nll

        self._loss.backward()
        self.optimizer.step()
        
        return self._loss.item()

    def x2z(self, batch):
        x = nn.Parameter(torch.FloatTensor(batch))
        mean = self._encode(x, self._latent_dim)
        return mean.data.numpy().reshape(-1, self._latent_dim)

    def z2x(self, z):
        x = self._decode(z)
        return x.data.numpy()

    def sample(self, z):
        x = self._decode(z)
        return x.data.numpy()

    def reconstruct(self, batch):
        x = nn.Parameter(torch.FloatTensor(batch))
        sample = self._decode(x)
        return sample.data.numpy()

    def save_encoder(self, path, prefix="vae/encoder"):
        encoder_vars = []
        for n, p in self.named_parameters():
            if "encoder" in n:
                encoder_vars.append((prefix + re.sub("encoder", "", n)).replace(".", "/"))
        
        encoder_dict = {}
        for var_str in encoder_vars:
            encoder_dict[var_str] = self.state_dict()[var_str]
        
        torch.save(encoder_dict, path)
    
    def save_decoder(self, path, prefix="vae/decoder"):
        decoder_vars = []
        for n, p in self.named_parameters():
            if "decoder" in n:
                decoder_vars.append((prefix + re.sub("decoder", "", n)).replace(".", "/"))
        
        decoder_dict = {}
        for var_str in decoder_vars:
            decoder_dict[var_str] = self.state_dict()[var_str]
        
        torch.save(decoder_dict, path)


class MMD_CVAE(nn.Module):

    def __init__(self, latent_dim, batch_size, encoder, decoder,
                 observation_dim=784,
                 learning_rate=1e-4,
                 optimizer=torch.optim.RMSprop,
                 observation_distribution="Bernoulli", # or Gaussian
                 observation_std=0.01):

        super(MMD_CVAE, self).__init__()

        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
        self._observation_dim = observation_dim
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._build_graph()

    def _build_graph(self):

        self.x = nn.Parameter(torch.Tensor())
        self.y = nn.Parameter(torch.Tensor())

        self.encoder = self._encode()
        self.decoder = self._decode()

        self.z = self.encoder(self.x, self._latent_dim)
        self.sample = self.decoder(self.z, self._observation_dim / 2)

        self.loss_mmd = self._compute_mmd(torch.normal(0, 1, size=(self.z.size(0), self._latent_dim)), self.z)
        self.loss_nll = torch.sum(nn.MSELoss()(self.y, self.sample))

        self._loss = self.loss_mmd + self.loss_nll

        variables = self.parameters()
        de_var_list = []
        for v in variables:
            if "decoder" in v.name:
                de_var_list.append(v)
        en_var_list = []
        for v in variables:
            if "encoder" in v.name:
                en_var_list.append(v)

        self.optimizer_de = self._optimizer(self._learning_rate)
        self.optimizer_en = self._optimizer(self._learning_rate)

    def _compute_mmd(self, x, y):
        x_kernel = self._compute_kernel(x, x)
        y_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def _compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean(torch.square(tiled_x - tiled_y), dim=2) / dim)

    def update(self, x, y):
        self.optimizer_de.zero_grad()
        loss = self._loss.backward(retain_graph=True)
        self.optimizer_de.step()

        if 1:
            self.optimizer_en.zero_grad()
            loss = self._loss.backward()
            self.optimizer_en.step()

        return loss.item()

    def x2z(self, x):
        self.eval()
        with torch.no_grad():
            mean = self.encoder(x, self._latent_dim)

        return mean.numpy().reshape(-1, self._latent_dim)

    def z2x(self, z):
        self.eval()
        with torch.no_grad():
            x = self.decoder(z, self._observation_dim / 2)
        return x.numpy()

    def sample(self, z):
        self.eval()
        with torch.no_grad():
            x = self.decoder(z, self._observation_dim / 2)
        return x.numpy()

    def generate(self, x):
        self.eval()
        with torch.no_grad():
            sample = self.decoder(x)
        return sample

    def load_pretrained_encoder(self, path):
        encoder_variables = []
        for v in self.parameters():
            if "encoder" in v.name:
                encoder_variables.append(v)
        saver = torch.load(path, map_location='cpu')
        for key, value in saver.items():
            if key in encoder_variables:
                encoder_variables[key] = value
        self.load_state_dict(encoder_variables)

    def load_pretrained_decoder(self, path):
        decoder_variables = []
        for v in self.parameters():
            if "decoder" in v.name:
                decoder_variables.append(v)
        saver = torch.load(path, map_location='cpu')
        for key, value in saver.items():
            if key in decoder_variables:
                decoder_variables[key] = value
        self.load_state_dict(decoder_variables)

    def save_encoder(self, path, prefix="in/encoder"):
        variables = self.parameters()
        var_dict = {}
        for v in variables:
            if "encoder" in v.name:
                name = prefix+re.sub("gen/encoder", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)
        torch.save(var_dict, path)

    def save_decoder(self, path, prefix="in/generator"):
        variables = self.parameters()
        var_dict = {}
        for v in variables:
            if "decoder" in v.name:
                name = prefix+re.sub("gen/decoder", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)
        torch.save(var_dict, path)