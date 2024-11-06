##
# @file db.py
# @author Keren Zhu
# @date Feb 2019
# @brief database and dataset for the routing guide
#

import math
import glob
import sys
import cv2
import util
import os.path
import torch
import numpy as np
from scipy import misc
import imageio

from tqdm import tqdm, trange

class Dataset:
    def __init__(self, dimension_):
        self.dimension = dimension_
        self.n_ch = 2 # One for placement, one for the needed pin

    def num_labeled(self):
        return len(self.in_labeled)
    
    def num_unlabeled(self):
        return len(self.in_unlabeled)

    def read_data_labeled(self, bench_path = ""):
        in_list = []
        out_list = []
        print("*" * 20 + "read data labeled" + "*" * 20)

        # channels are 
        if bench_path != "":
            for bench in tqdm(glob.glob(bench_path + "/*")):
                in_chs = np.zeros((self.dimension, self.dimension, self.n_ch))
                out_chs = np.zeros((self.dimension, self.dimension, self.n_ch))
                for ch in  range(0, self.n_ch):
                    in_img = bench + "/pin" + str(ch) + ".png"
                    out_img = bench + "/route" + str(ch) + ".png"
                    in_chs[:,:,ch] = imageio.imread(in_img)
                    out_chs[:,:,ch] = imageio.imread(out_img)
                in_list.append(in_chs)
                out_list.append(out_chs)
        self.in_labeled = np.array(in_list)
        self.out_labeled = np.array(out_list)

    def read_data_labeled_net(self, bench_path=""):
        in_list = []
        out_list = []
        print("*" * 20 + "read data labeled net" + "*" * 20)

        if bench_path != "":
            for bench in tqdm(glob.glob(bench_path + "/*")):
                in_chs = np.zeros((self.dimension, self.dimension, self.n_ch))
                out_chs = np.zeros((self.dimension, self.dimension, self.n_ch))
                placement_img = bench + "/place.png"
                idx = 0
                while 1:
                    pin_img = bench + "/net_pin" + str(idx) + ".png"
                    route_img = bench + "/net_route" + str(idx) + ".png"
                    if not (os.path.isfile(pin_img)):
                        break
                    in_chs[:,:,0] = imageio.imread(placement_img) # pins for all placement
                    in_chs[:,:,1] = imageio.imread(pin_img) # pins for one net
                    out_chs[:,:,1] = imageio.imread(route_img)
                    in_list.append(in_chs)
                    out_list.append(out_chs)
                    idx += 1
        self.in_labeled = np.array(in_list)
        self.out_labeled = np.array(out_list)

    def read_data_unlabeled(self, bench_path=""):
        in_list = []
        print("*" * 20 + "read data unlabeled net" + "*" * 20)

        if bench_path != "":
            for bench in tqdm(glob.glob(bench_path + "/*")):
                in_chs = np.zeros((self.dimension, self.dimension, self.n_ch))
                placement_img = bench + "/place.png"
                num_pin = 0
                for pin_img in glob.glob(bench+"/*"):
                    if pin_img[pin_img.rfind('/') + 1 :] == "place.png":
                        continue
                    if num_pin > 20:# do not overly unbalance the data set
                        break
                    num_pin += 1
                    in_chs[:,:,0] = imageio.imread(placement_img) # pins for all placement
                    in_chs[:,:,1] = imageio.imread(pin_img) # pins for one net
                    in_list.append(in_chs)
        self.in_unlabeled = np.array(in_list)
        print("Number of unlabeled data ", len(self.in_unlabeled))

    def read_data_test(self, bench = ""):
        in_list = []
        print("*" * 20 + "read data test" + "*" * 20)

        if (bench == ""):
            self.in_test = np.array(in_list)
        else:
            # channels are
            in_chs = np.zeros((self.dimension, self.dimension, self.n_ch))
            for ch in trange(0, self.n_ch):
                in_img = bench + "/pin" + str(ch) + ".png"
                in_chs[:,:,ch] = imageio.imread(in_img)
            in_list.append(in_chs)
            self.in_test = np.array(in_list)

    def image_standardization(self):
        #self.train_mean = np.array(self.in_unlabeled).mean(axis=(0,1,2))
        #self.train_var = np.array(self.in_unlabeled).std(axis=(0,1,2))
        self.in_unlabeled = self.in_unlabeled.astype(np.float32)
        self.in_labeled = self.in_labeled.astype(np.float32)
        self.out_labeled = self.out_labeled.astype(np.float32)
        self.in_test = self.in_test.astype(np.float32)
        for i in range(2):
            #self.in_unlabeled[:,:,:,i] = (self.in_unlabeled[:,:,:,i] - self.train_mean[i]) / self.train_var[i]
            if len(self.in_unlabeled) > 0:
                self.in_unlabeled[:,:,:,i] = (self.in_unlabeled[:,:,:,i] - 0) / 255
            if len(self.in_labeled) > 0:
                self.in_labeled[:,:,:,i] = (self.in_labeled[:,:,:,i] - 0) / 255
            if len(self.out_labeled) > 0:
                self.out_labeled[:,:,:,i] = (self.out_labeled[:,:,:,i] - 0) / 255
            if len(self.in_test) > 0:
                self.in_test[:,:,:,i] = (self.in_test[:,:,:,i] - 0) / 255

    def gaussian_blur(self, pixel):
        """
        for img in self.in_labeled:
            for ch in range(0, self.n_ch):
                img[:,:,ch] = cv2.GaussianBlur(img[:,:,ch], (pixel, pixel), 0)
        """
        print("*" * 20 + "gaussian_blur" + "*" * 20)
        for img in tqdm(self.out_labeled):
            for ch in range(0, self.n_ch):
                img[:,:,ch] = cv2.GaussianBlur(img[:,:,ch], (pixel, pixel), 0)
        """
        for img in self.in_unlabeled:
            for ch in range(0, self.n_ch):
                img[:,:,ch] = cv2.GaussianBlur(img[:,:,ch], (pixel, pixel), 0)
        """

    def gaussian_blur_pin(self, pixel):
        for img in self.in_labeled:
            for ch in range(0, self.n_ch):
                img[:,:,ch] = cv2.GaussianBlur(img[:,:,ch], (pixel, pixel), 0)
        for img in self.in_unlabeled:
            for ch in range(0, self.n_ch):
                img[:,:,ch] = cv2.GaussianBlur(img[:,:,ch], (pixel, pixel), 0)
        for img in self.in_test:
            for ch in range(0, self.n_ch):
                img[:,:,ch] = cv2.GaussianBlur(img[:,:,ch], (pixel, pixel), 0)

    def reduct_pin(self, ch_idx):
        for img_idx in range(0, len(self.in_labeled)):
            self.out_labeled[img_idx][:,:,ch_idx] = np.subtract(self.out_labeled[img_idx][:,:,ch_idx], np.divide(self.in_labeled[img_idx][:,:,ch_idx], 2))
    
    def add_labeled_data_to_unlabeled_set(self):
        np.append(self.in_unlabeled, self.in_labeled)
        #for data in self.in_labeled:
           # np.append(self.in_unlabeled.append, data)
    
    def next_unlabeled_batch(self, num):
        idx = np.arange(0, len(self.in_unlabeled))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [np.reshape(self.in_unlabeled[ i], 64*64*2) for i in idx]
        data_shuffle = np.array(data_shuffle)
        return torch.asarray(data_shuffle)

    def next_unlabeled_batch(self, num):
        idx = np.arange(0, len(self.in_unlabeled))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [np.reshape(self.in_unlabeled[ i], 64*64*2) for i in idx]
        data_shuffle = np.array(data_shuffle)
        return torch.asarray(data_shuffle)

    def next_unlabeled_batch_2ch(self, num):
        idx = np.arange(0, len(self.in_unlabeled))
        rand_idx = np.arange(0, len(self.in_unlabeled))
        np.random.shuffle(idx)
        np.random.shuffle(rand_idx)
        idx = idx[:num]
        rand_idx = rand_idx[:num]
        data_shuffle = [np.reshape(np.concatenate((self.in_unlabeled[idx[i]], np.expand_dims(self.in_unlabeled[rand_idx[i],:,:,1], -1)), axis=-1), 64*64*3) for i in range(len(idx))]
        data_shuffle = np.array(data_shuffle)
        return torch.asarray(data_shuffle)

    def next_labeled_batch(self, num):
        idx = np.arange(0, len(self.in_labeled))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [np.reshape(self.in_labeled[ i], 64*64*2) for i in idx]
        label_shuffle = [np.reshape(self.out_labeled[i][:,:,1], 64*64*1) for i in idx]
        data_shuffle = np.array(data_shuffle)
        label_shuffle = np.array(label_shuffle)
        return torch.asarray(data_shuffle), torch.asarray(label_shuffle)

    def test_input(self):
        return torch.asarray(np.reshape(self.in_test[0], (1, 64 * 64 *2)))


    def data_augment(self):
        print("*" * 20 + "data_augment" + "*" * 20)
        num_image = self.num_labeled()

        # flip horizontal
        print("*" * 20 + "labeled data: flip horizontal" + "*" * 20)
        for idx in trange(0, num_image):
            self.in_labeled = util.np_append(self.in_labeled, cv2.flip(self.in_labeled[idx], 0))
            self.out_labeled = util.np_append(self.out_labeled, cv2.flip(self.out_labeled[idx], 0))

        """
        # rotate 180
        for idx in range(0, num_image):
            self.in_labeled = util.np_append(self.in_labeled, util.rotate_image(self.in_labeled[idx], 180))
            self.out_labeled = util.np_append(self.out_labeled, util.rotate_image(self.out_labeled[idx], 180))
        """

        # flip vertical
        print("*" * 20 + "labeled data: flip vertical" + "*" * 20)
        for idx in trange(0, num_image):
            self.in_labeled = util.np_append(self.in_labeled, cv2.flip(self.in_labeled[idx], 1))
            self.out_labeled = util.np_append(self.out_labeled, cv2.flip(self.out_labeled[idx], 1))
            self.in_labeled = util.np_append(self.in_labeled, cv2.flip(self.in_labeled[-1], 0))#< added
            self.out_labeled = util.np_append(self.out_labeled, cv2.flip(self.out_labeled[-1], 0))#< added
        """
        #transpose and 90
        for idx in range(0, num_image):
            self.in_labeled = util.np_append(self.in_labeled, util.rotate_image(self.in_labeled[idx], 90))
            self.out_labeled = util.np_append(self.out_labeled, util.rotate_image(self.out_labeled[idx], 90))
            self.in_labeled = util.np_append(self.in_labeled, cv2.flip(self.in_labeled[-1], 0))
            self.out_labeled = util.np_append(self.out_labeled, cv2.flip(self.out_labeled[-1], 0))
        #transverse and 270
        for idx in range(0, num_image):
            self.in_labeled = util.np_append(self.in_labeled, util.rotate_image(self.in_labeled[idx], 270))
            self.out_labeled = util.np_append(self.out_labeled, util.rotate_image(self.out_labeled[idx], 270))
            self.in_labeled = util.np_append(self.in_labeled, cv2.flip(self.in_labeled[-1], 0))
            self.out_labeled = util.np_append(self.out_labeled, cv2.flip(self.out_labeled[-1], 0))
        """

        #Unsupervised
        num_image = self.num_unlabeled()

        # flip horizontal
        print("*" * 20 + "unlabeled data: flip horizontal" + "*" * 20)
        for idx in trange(0, num_image):
            self.in_unlabeled = util.np_append(self.in_unlabeled, cv2.flip(self.in_unlabeled[idx], 0))
        """
        # rotate 180
        for idx in range(0, num_image):
            self.in_unlabeled = util.np_append(self.in_unlabeled, util.rotate_image(self.in_unlabeled[idx], 180))
        """

        # flip vertical
        print("*" * 20 + "labeled data: flip vertical" + "*" * 20)
        for idx in trange(0, num_image):
            self.in_unlabeled = util.np_append(self.in_unlabeled, cv2.flip(self.in_unlabeled[idx], 1))
            self.in_unlabeled = util.np_append(self.in_unlabeled, cv2.flip(self.in_unlabeled[-1], 0)) #< added
        """
        #transpose and 90
        for idx in range(0, num_image):
            self.in_unlabeled = util.np_append(self.in_unlabeled, util.rotate_image(self.in_unlabeled[idx], 90))
            self.in_unlabeled = util.np_append(self.in_unlabeled, cv2.flip(self.in_unlabeled[-1], 0))
        #transverse and 270
        for idx in range(0, num_image):
            self.in_unlabeled = util.np_append(self.in_unlabeled, util.rotate_image(self.in_unlabeled[idx], 270))
            self.in_unlabeled = util.np_append(self.in_unlabeled, cv2.flip(self.in_unlabeled[-1], 0))
        """ 
        print ("labeled data # ", self.in_labeled.shape)