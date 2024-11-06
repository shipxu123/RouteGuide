##
# @file train.py
# @author Keren Zhu
# @date Feb 2019
#

from PIL import Image
from scipy import misc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def save_to_grayscale_png(np_list, fname):
    misc.imsave(fname, np_list)

def normalize_int_0_to_range(np_list, scale_range):
    _max = np.max(np_list)
    _min = np.min(np_list)
    return ((np_list - _min) * scale_range / (_max - _min)).astype(int)

def np_append(array, element):
    return np.concatenate([array, np.array(element[None, :])])

def export_grayscale_image_text(np_list, fname):
    scale_range = 255
    np_list = cv2.GaussianBlur(np_list, (7, 7), 0)
    int_list = normalize_int_0_to_range(np_list, scale_range)
    with open(fname, "w") as f:
        # # of rows
        f.write(str(len(np_list)))
        f.write(' ')
        # # of cols
        f.write(str(len(np_list[0])))
        f.write(' ')
        # sclae range
        f.write(str(scale_range))
        f.write('\n')
        for rowIdx in range(0, len(np_list)):
            for colIdx in range(0, len(np_list[0])):
                gray = int_list[colIdx][rowIdx]
                f.write(str(gray))
                f.write('\n')

def draw_five_reconstruct_samples(img_size, golden, generated):
    n_examples = len(golden)
    if (n_examples > 5):
        n_examples = 5

    fig, axs = plt.subplots(4, n_examples, figsize=(n_examples, 4))
    for example_i in range(n_examples):
        xs_img = np.reshape(golden[example_i], [img_size, img_size, 2])
        ys_img = np.reshape(generated[example_i], [img_size, img_size, 2])
        axs[0][example_i].imshow(
                xs_img[:,:,0])
        axs[1][example_i].imshow(
                xs_img[:,:, 1])
        axs[2][example_i].imshow(
                ys_img[:,:,0])
        axs[3][example_i].imshow(
                ys_img[:,:,1])
        """
        axs[2][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (dimension * dimension,)) ,
    #            (dimension, dimension)))
        """
    figname = "./fig/"+ "valid" + ".png"
    plt.savefig(figname)


def draw_five_generated_samples(img_size, golden, label, generated):
    n_examples = len(golden)
    if (n_examples > 5):
        n_examples = 5
    if n_examples > 1:
        fig, axs = plt.subplots(4, n_examples, figsize=(n_examples, 4))
    else:
        fig, axs = plt.subplots(4, 2, figsize=(2, 4))
    for example_i in range(n_examples):
        xs_img = np.reshape(golden[example_i], [img_size, img_size, 2])
        label_img = np.reshape(label[example_i], [img_size, img_size, 1])
        import pdb; pdb.set_trace()
        ys_img = np.reshape(generated[example_i], [img_size, img_size, 1])
        axs[0][example_i].imshow(
                xs_img[:,:,0])
        axs[1][example_i].imshow(
                xs_img[:,:, 1])
        axs[2][example_i].imshow(
                label_img[:,:, 0])
        axs[3][example_i].imshow(
                ys_img[:,:,0])
        """
        axs[2][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (dimension * dimension,)) ,
                (dimension, dimension)))
        """
    figname = "./fig/"+ "valid" + ".png"
    plt.savefig(figname)

def export_test_image(img_size, golden, generated):
    xs_img = np.reshape(golden[0], [img_size, img_size, 2])
    ys_img = np.reshape(generated[0], [img_size, img_size, 1])
    plt.imsave('./fig/ch0.png', xs_img[:,:,0], cmap = cm.gray)
    plt.imsave('./fig/ch1.png', xs_img[:,:,1], cmap = cm.gray)
    plt.imsave('./fig/out.png', ys_img[:,:,0], cmap = cm.gray)

def draw_five_test(img_size, golden, generated):
    n_examples = len(golden)
    if (n_examples > 5):
        n_examples = 5
    if n_examples > 1:
        fig, axs = plt.subplots(3, n_examples, figsize=(n_examples, 3))
    else:
        fig, axs = plt.subplots(3, 2, figsize=(2, 3))

    for example_i in range(n_examples):
        xs_img = np.reshape(golden[example_i], [img_size, img_size, 2])
        ys_img = np.reshape(generated[example_i], [img_size, img_size, 1])
        axs[0][example_i].imshow(
                xs_img[:,:,0])
        axs[1][example_i].imshow(
                xs_img[:,:, 1])
        axs[2][example_i].imshow(
                ys_img[:,:,0])
        """
        axs[2][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (dimension * dimension,)) ,
                (dimension, dimension)))
        """
    figname = "./fig/"+ "valid" + ".png"
    plt.savefig(figname)
