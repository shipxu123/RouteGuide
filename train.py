##
# @file train.py
# @author Keren Zhu
# @date Feb 2019
#

import sys
import torch
import imageio

import numpy as np
import tensorflow as tf


import util
import coders.ae_coding as aec

from db import Dataset

from models.vae import VAE
from models.ae import Autoencoder
from models.mmd_vae import MMD_VAE
import matplotlib.pyplot as plt

def prepare_data(ds):
    ds.read_data_unlabeled("../unsupervised_bench")
    ds.read_data_labeled("../bench5")
    ds.read_data_test("")
    ds.data_augment()
    ds.gaussian_blur_pin(5)
    ds.image_standardization()
    ds.add_labeled_data_to_unlabeled_set()
    print("Data preparation finished \n")

def train_unlabeled_ae(ds):
    re_flags = tf.compat.v1.flags # flags for reconstructing task
    re_flags.DEFINE_integer("latent_dim", 32 , "dimension of latent space")
    re_flags.DEFINE_integer("obs_dim", 64*64*2, "dimension of observation space")
    re_flags.DEFINE_integer("batch_size", 64, "Batch size")
    re_flags.DEFINE_integer("epochs", 1500, "number of epoch")
    re_flags.DEFINE_integer("updates_per_epoch", 1 ,"mini-batch" )
    FLAGS = re_flags.FLAGS

    kwargs = {
            'latent_dim' : FLAGS.latent_dim,
            'batch_size' : FLAGS.batch_size,
            'observation_dim': FLAGS.obs_dim,
            'encoder': aec.ConvEncoder,
            'decoder': aec.ConvDecoder,
            'optimizer': torch.optim.Adam
    }
    ae = Autoencoder(**kwargs)
    ae = ae.cuda()

    for epoch in range(FLAGS.epochs):
        training_loss = 0.0

        for _ in range(FLAGS.updates_per_epoch):
            #x = ds.next_unlabeled_batch(FLAGS.batch_size)
            x = []
            for idx in range(FLAGS.batch_size):
                x.append(torch.reshape(ds.in_unlabeled[0], 64*64*2))

            x = torch.Tensor(x)
            x = x.cuda()
            loss = ae.update(x)
            training_loss += loss

        training_loss /= FLAGS.updates_per_epoch
        s = "Loss: {: .4f}".format(training_loss)
        print(epoch, s)

    x = []
    for idx in range(FLAGS.batch_size):
        x.append(np.reshape(ds.in_unlabeled[0], 64*64*2))

    test = ds.next_unlabeled_batch(FLAGS.batch_size)
    sample = ae.reconstruct(x)
    util.draw_five_reconstruct_samples(64, x, sample[0])

def train_unlabeled_vae_2ch(ds):
    re_flags = tf.compat.v1.flags # flags for reconstructing task
    re_flags.DEFINE_integer("latent_dim", 64, "dimension of latent space")
    re_flags.DEFINE_integer("obs_dim", 64*64*2, "dimension of observation space")
    re_flags.DEFINE_integer("batch_size", 64, "Batch size")
    re_flags.DEFINE_integer("epochs", 2000, "number of epoch")
    re_flags.DEFINE_integer("updates_per_epoch", 100 ,"mini-batch" )
    FLAGS = re_flags.FLAGS

    kwargs = {
            'latent_dim' : FLAGS.latent_dim,
            'batch_size' : FLAGS.batch_size,
            'observation_dim': FLAGS.obs_dim,
            'encoder': aec.ConvEncoder_2ch,
            'decoder': aec.ConvDecoder_2ch,
            'observation_distribution': 'Gaussian',
            'image_ch_dim' : 3
            }
    vae = VAE(**kwargs)

    for epoch in range(FLAGS.epochs):
        training_loss = 0

        for _ in range(FLAGS.updates_per_epoch):
            x = ds.next_unlabeled_batch_2ch(FLAGS.batch_size)
            loss = vae.update(x)
            training_loss += loss

        training_loss /= FLAGS.updates_per_epoch
        s = "Loss: {: .4f}".format(training_loss)
        print(epoch, s)

    test = ds.next_unlabeled_batch(FLAGS.batch_size)
    sample = vae.reconstruct(test)
    util.draw_five_reconstruct_samples(64, test, sample[0])

    vae.save_encoder("weights/vae_2ch_fix/encoder", prefix="gen/encoder")


def train_unlabeled_vae(ds):
    re_flags = tf.compat.v1.flags # flags for reconstructing task
    re_flags.DEFINE_integer("latent_dim", 32, "dimension of latent space")
    re_flags.DEFINE_integer("obs_dim", 64*64*2, "dimension of observation space")
    re_flags.DEFINE_integer("batch_size", 64, "Batch size")
    re_flags.DEFINE_integer("epochs", 5000, "number of epoch")
    # re_flags.DEFINE_integer("epochs", 5, "number of epoch")
    re_flags.DEFINE_integer("updates_per_epoch", 100 ,"mini-batch" )
    FLAGS = re_flags.FLAGS

    kwargs = {
            'latent_dim' : FLAGS.latent_dim,
            'batch_size' : FLAGS.batch_size,
            'observation_dim': FLAGS.obs_dim,
            'encoder': aec.ConvEncoder,
            'decoder': aec.ConvDecoder,
            'observation_distribution': 'Gaussian'
    }
    vae = VAE(**kwargs)
    vae = vae.cuda()

    for epoch in range(FLAGS.epochs):
        training_loss = 0

        for _ in range(FLAGS.updates_per_epoch):
            x = ds.next_unlabeled_batch(FLAGS.batch_size)
            x = x.cuda()
            loss = vae.update(x)
            training_loss += loss

        training_loss /= FLAGS.updates_per_epoch
        s = "Loss: {: .4f}".format(training_loss)
        print(epoch, s)

    test = ds.next_unlabeled_batch(FLAGS.batch_size)
    test = test.cuda()
    sample = vae.reconstruct(test)
    util.draw_five_reconstruct_samples(64, test.cpu().numpy(), sample)

    vae.save_encoder("weights/vae_fix_32/encoder.pt", prefix="gen/encoder")
    vae.save_decoder("weights/vae_fix_32/decoder.pt", prefix="gen/decoder")


def train_unlabeled_mmd_vae(ds):
    re_flags = tf.compat.v1.flags # flags for reconstructing task
    re_flags.DEFINE_integer("latent_dim", 32, "dimension of latent space")
    re_flags.DEFINE_integer("obs_dim", 64*64*2, "dimension of observation space")
    re_flags.DEFINE_integer("batch_size", 64, "Batch size")
    re_flags.DEFINE_integer("epochs", 5500, "number of epoch")
    re_flags.DEFINE_integer("updates_per_epoch", 100 ,"mini-batch" )
    FLAGS = re_flags.FLAGS

    kwargs = {
            'latent_dim' : FLAGS.latent_dim,
            'batch_size' : FLAGS.batch_size,
            'observation_dim': FLAGS.obs_dim,
            'encoder': aec.ConvEncoder,
            'decoder': aec.ConvDecoder
            }
    vae = MMD_VAE(**kwargs)

    for epoch in range(FLAGS.epochs):
        training_loss = 0
        mmd_loss = 0
        nll_loss = 0

        for _ in range(FLAGS.updates_per_epoch):
            x = ds.next_unlabeled_batch(FLAGS.batch_size)
            loss, mmd, nll  = vae.update(x)
            training_loss += loss
            mmd_loss += mmd
            nll_loss += nll

        training_loss /= FLAGS.updates_per_epoch
        mmd_loss /= FLAGS.updates_per_epoch
        nll_loss /= FLAGS.updates_per_epoch
        s = "Loss: {: .4f} mmd: {: .4f} nll: {: .4f}".format(training_loss, mmd_loss, nll_loss)
        print(epoch, s)

    test = ds.next_unlabeled_batch(FLAGS.batch_size)
    sample = vae.reconstruct(test)
    util.draw_five_reconstruct_samples(64, test, sample)

    vae = vae.cpu()
    vae.save_encoder("weights/mmd_vae/encoder.pt", prefix="gen/encoder")
    vae.save_decoder("weights/mmd_vae/decoder.pt", prefix="gen/decoder")

def main(mode):
    ds = Dataset(64)
    prepare_data(ds)
    ds.next_unlabeled_batch_2ch(10)
    if mode == "mmd_vae":
        train_unlabeled_mmd_vae(ds)
    else:
        train_unlabeled_vae(ds)


if __name__ == '__main__':
    main(sys.argv[1])
