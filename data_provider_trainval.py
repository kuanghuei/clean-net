# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""Data providers for training and validation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math
import os
import sys

import numpy as np
import tensorflow as tf


def get_data_batcher(mode, opt):
    ref_data = np.load(os.path.join(opt.data_dir, "ref.npy"))
    if mode == 'train':
        query_data = np.load(os.path.join(opt.data_dir, "train.npy"))
        return VerifiedTrainValDataBatcher(ref_data, query_data, opt.num_ref, opt.img_dim, is_training=True)
    elif mode == 'val':
        query_data = np.load(os.path.join(opt.data_dir, "val.npy"))
        return VerifiedTrainValDataBatcher(ref_data, query_data, opt.num_ref, opt.img_dim, is_training=False)
    elif mode == 'unverified':
        unverified_data = np.load(os.path.join(opt.data_dir, "all.npy"))
        return UnverifiedDataBatcher(ref_data, unverified_data, opt.num_ref, opt.img_dim)
    else:
        raise ValueError('Mode unknown %s' % mode)


class VerifiedTrainValDataBatcher():
    """Data batcher for training/validation data with verification labels"""

    def __init__(self, ref_data, query_data, num_ref, img_dim, is_training, rand_seed=31, np_rand_seed=123):
        """Basic setup

        Args:
            ref_data: numpy arrary of reference data
                      (num_class, num_ref, img_dim)
            query_data: numpy array of query data with verification labels
                        (num_sample, img_dim)
            num_ref: number of reference embeddings
            img_dim: dimension of image feature
            is_training: True - training model / False - inference model
            rand_seed: random seed
            np_rand_seed: numpy random seed
        """
        self.ref_data = ref_data
        self.query_data = query_data
        self.num_ref = num_ref
        self.img_dim = img_dim
        self.is_training = is_training
        self.data_size = query_data.shape[0]
        self.i = 0
        random.seed(rand_seed)
        np.random.seed(np_rand_seed)

        if self.is_training:
            np.random.shuffle(self.query_data)


    def reset(self):
        """reset data provider"""
        if self.is_training:
            np.random.shuffle(self.query_data)
        self.i = 0


    def get_batch(self, batch_size):
        """Get a batch of training data

        Args:
            batch_size: batch size
        Returns:
            batch_class_id: class id based on the order of class list (numpy array)
            batch_vlabel: verification label (numpy array)
            batch_query: queries (numpy array)
            batch_vflag: verification flags indicating a sample is for supervised(1) 
                         or unsupervised(0) training (numpy array)
            batch_ref: reference embeddings (numpy array)
        """

        # data pointer manipulation
        if self.i + batch_size < self.data_size:
            query_samples = self.query_data[self.i:self.i+batch_size,:]
            self.i += batch_size
        elif self.i + batch_size == self.data_size:
            if self.is_training:
                np.random.shuffle(self.query_data)
            query_samples = self.query_data[:batch_size,:]
            self.i = batch_size
        else:
            part1 = self.query_data[self.i:self.data_size,:]
            if self.is_training:
                np.random.shuffle(self.query_data)
            query_samples = np.concatenate((part1, self.query_data[:(self.i+batch_size-self.data_size),:]), axis=0)
            self.i += batch_size - self.data_size

        # split data
        batch_class_id = query_samples[:,-2]
        batch_vlabel = query_samples[:,-1]
        batch_query = query_samples[:,:-2]
        batch_vflag = np.ones([batch_size])

        # get corresponding reference set
        batch_ref = []
        for i in range(batch_size):
            batch_ref.append([self.ref_data[int(batch_class_id[i])][j] for j in range(self.num_ref)])
        batch_ref = np.stack(batch_ref, axis=0)

        return batch_class_id, batch_vlabel, batch_query, batch_vflag, batch_ref


class UnverifiedDataBatcher():
    """Data batcher that randomly picks training data without verification labels"""
    def __init__(self, ref_data, unverified_data, num_ref, img_dim, rand_seed=31, np_rand_seed=123):
        """Basic setup

        Args:
            ref_data: numpy arrary of reference data
                      (num_class, num_ref, img_dim)
            unverified_data: numpy array of query data without verification labels
                        (num_class, num_sample_inclass, img_dim)
            num_ref: number of reference embeddings
            img_dim: dimension of image feature
            rand_seed: random seed
            np_rand_seed: numpy random seed
        Returns:
            phi_s: reference set vector
            v_q: query image feature (batch_size, img_dim)
            phi_q: query vector
            v_qr: reconstructed v_q
        """

        self.ref_data = ref_data
        self.unverified_data = unverified_data
        self.num_ref = num_ref
        self.img_dim = img_dim
        random.seed(rand_seed)
        np.random.seed(np_rand_seed)


    def reset():
        """do nothing"""
        pass


    def get_batch(self, batch_size):
        """Get a batch of training data

        Args:
            batch_size: batch size
        Returns:
            batch_class_id: class id based on the order of class list (numpy array)
            batch_vlabel_dummy: dummy verification label (not used)
            batch_query: queries (numpy array)
            batch_vflag: verification flags indicating a sample is for supervised(1) 
                         or unsupervised(0) training (numpy array)
            batch_ref: reference embeddings (numpy array)
        """

        # randomly pick classes
        batch_class_id = np.random.randint(len(self.unverified_data), size=batch_size)
        batch_query = np.zeros([batch_size, self.img_dim])
        batch_v_flag = np.zeros([batch_size])
        for i in range(batch_size):
            query_sample = random.choice(self.unverified_data[batch_class_id[i]])
            batch_query[i,:] = query_sample

        # get corresponding reference set
        batch_ref = []
        for i in range(batch_size):
            batch_ref.append([self.ref_data[int(batch_class_id[i])][j] for j in range(self.num_ref)])
        batch_ref = np.stack(batch_ref, axis=0)
        
        batch_vlabel_dummy = np.zeros([batch_size])
        return batch_class_id, batch_vlabel_dummy, batch_query, batch_v_flag, batch_ref
