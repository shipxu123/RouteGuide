##
# @file generator_dif.py
# @author Peng XU
# @date Feb 2023
#
import os
import sys
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import util
import coders.ae_coding as aec
import matplotlib.pyplot as plt

from db import Dataset

from models.vae import VAE_GEN
from models.mmd_vae import MMD_CVAE

def prepare_data(ds, test_path):
    ds.read_data_unlabeled("")
    ds.read_data_labeled("")
    #ds.data_augment()
    ds.read_data_test(test_path)
    ds.image_standardization()
    ds.gaussian_blur_pin(5)
    #ds.gaussian_blur(7)
    #ds.add_labeled_data_to_unlabeled_set()
    print("Data preparation finished \n")


def generate_vae(ds):
    re_flags = tf.compat.v1.flags # flags for reconstructing task
    re_flags.DEFINE_integer("latent_dim", 32, "dimension of latent space")
    re_flags.DEFINE_integer("obs_dim", 64*64*2, "dimension of observation space")
    re_flags.DEFINE_integer("batch_size", 8, "Batch size")
    re_flags.DEFINE_integer("epochs", 100, "number of epoch")
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
    # state_dict = torch.load("./weights/vae_fix_32/encoder.pt")
    # vae.load_state_dict(state_dict)
    vae.load_pretrained_encoder("weights/vae_gen_dif_fix_32_test/encoder")
    vae.load_pretrained_decoder("weights/vae_gen_dif_fix_32_test/decoder")
    vae = vae.cuda()

    test = ds.test_input()
    test = test.cuda()
    sample = vae.generate(test)
    util.draw_five_test(64, test.cpu(), sample)
    util.export_test_image(64, test.cpu(), sample)

    realname = os.path.join(sys.argv[1], "guide.txt")
    real_img = np.reshape(sample[0], [64, 64])
    util.export_grayscale_image_text(real_img, realname)


def generate_mmd_vae(ds):
    re_flags = tf.flags # flags for reconstructing task
    re_flags.DEFINE_integer("latent_dim", 16, "dimension of latent space")
    re_flags.DEFINE_integer("obs_dim", 64*64*2, "dimension of observation space")
    re_flags.DEFINE_integer("batch_size", 8, "Batch size")
    re_flags.DEFINE_integer("epochs", 100, "number of epoch")
    re_flags.DEFINE_integer("updates_per_epoch", 100 ,"mini-batch" )
    FLAGS = re_flags.FLAGS

    kwargs = {
            'latent_dim' : FLAGS.latent_dim,
            'batch_size' : FLAGS.batch_size,
            'observation_dim': FLAGS.obs_dim,
            'encoder': aec.conv_encoder,
            'decoder': aec.conv_decoder,
            'observation_distribution': 'Gaussian'
            }
    vae = VAE_GEN(**kwargs)

    vae.load_pretrained_encoder("weights/vae_gen_dif/encoder")
    vae.load_pretrained_decoder("weights/vae_gen_dif/decoder")

    test = ds.test_input()
    sample = vae.generate(test)
    util.draw_five_test(64, test, sample[0])


def main(bench_path):
    ds = Dataset(64)
    prepare_data(ds, bench_path)
    generate_vae(ds)


if __name__ == '__main__':
    main(sys.argv[1])