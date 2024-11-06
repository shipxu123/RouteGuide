##
# @file train.py
# @author Keren Zhu
# @date Feb 2019
#

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import util
import coders.ae_coding as aec

from db import Dataset
#import coders.vae_coding as vaec

from models.vae import VAE_GEN
from models.mmd_vae import MMD_CVAE


def prepare_data(ds):
    ds.read_data_unlabeled("")
    ds.read_data_labeled_net("../bench_pg")
    ds.read_data_test("")
    ds.data_augment()
    ds.image_standardization()
    ds.gaussian_blur(25)
    ds.gaussian_blur_pin(5)
    #ds.add_labeled_data_to_unlabeled_set()
    print("Data preparation finished \n")


def train_labeled_vae(ds):
    re_flags = tf.flags # flags for reconstructing task
    re_flags.DEFINE_integer("latent_dim", 32, "dimension of latent space")
    re_flags.DEFINE_integer("obs_dim", 64*64*2, "dimension of observation space")
    re_flags.DEFINE_integer("batch_size", 16, "Batch size")
    re_flags.DEFINE_integer("epochs", 800, "number of epoch")
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
    vae = VAE_GEN(**kwargs)
    vae = vae.cuda()

    vae.load_pretrained_encoder("./weights/vae_fix_32/encoder.pt")

    for epoch in range(FLAGS.epochs):
        training_loss = 0.0

        for _ in range(FLAGS.updates_per_epoch):
            x, y = ds.next_labeled_batch(FLAGS.batch_size)
            loss = vae.update_phrase1(x, y)
            training_loss += loss

        training_loss /= FLAGS.updates_per_epoch
        s = "Loss: {: .4f}".format(training_loss)
        print(epoch, s)

    print("Phrase 2: ")
    for epoch in range(FLAGS.epochs):
        training_loss = 0

        for _ in range(FLAGS.updates_per_epoch):
            x,y = ds.next_labeled_batch(FLAGS.batch_size)
            loss = vae.update_phrase2(x,y)
            training_loss += loss

        training_loss /= FLAGS.updates_per_epoch
        s = "Loss: {: .4f}".format(training_loss)
        print(epoch, s)

    test, label = ds.next_labeled_batch(FLAGS.batch_size)
    sample = vae.generate(test)
    util.draw_five_generated_samples(64, test, label, sample[0])

    vae.save_encoder("weights/vae_gen_pg_fix_32_test/encoder.pt", prefix="gen/encoder")
    vae.save_decoder("weights/vae_gen_pg_fix_32_test/decoder.pt", prefix="gen/decoder")


def main():
    ds = Dataset(64)
    prepare_data(ds)
    train_labeled_vae(ds)


if __name__ == '__main__':
    main()