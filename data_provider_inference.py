# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""Data providers for inference"""

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
    # load class names
    class_names = []
    with open(opt.class_list) as fp:
        for line in fp:
            class_names.append(line.strip())
    
    ref_data = np.load(os.path.join(opt.data_dir, "ref.npy"))
    
    if mode == 'inference':
        return InferenceDataBatcher(ref_data, opt.image_feature_list, class_names, opt.num_ref, opt.img_dim)
    else:
        raise ValueError('Mode unknown %s' % mode)


class InferenceDataBatcher():
    """Data batcher for training/validation data with verification labels"""

    def __init__(self, ref_data, image_feature_list, class_names, num_ref, img_dim, rand_seed=31, np_rand_seed=123):
        """Basic setup

        Args:
            ref_data: numpy arrary of reference data
                      (num_class, num_ref, img_dim)
            image_feature_list: path to the data source tsv, columns are:
                                [key, url, class_name, feature] or 
                                [key, class_name, feature]
            class_names: list of class names
            num_ref: number of reference embeddings
            img_dim: dimension of image feature
            rand_seed: random seed
            np_rand_seed: numpy random seed
        """
        self.fp = open(image_feature_list)
        self.ref_data = ref_data
        self.class_names = class_names
        self.class_names_to_ids = dict(zip(class_names, range(len(class_names))))
        self.num_ref = num_ref
        self.img_dim = img_dim
        self.eof = False

        random.seed(rand_seed)
        np.random.seed(np_rand_seed)


    def __del__(self):
        self.fp.close()


    def reset(self):
        """restart from the beginning"""
        self.fp.close()
        self.fp = open(self.source_file)
        self.eof = False


    def get_batch(self, batch_size):
        """Get a batch of training data

        Args:
            batch_size: batch size
        Returns:
            batch_class_id: class id based on the order of class list (numpy array)
                            (batch_size, )
            batch_query: queries (numpy array)
                         (batch_size, img_dim)
            batch_ref: reference embeddings (numpy array)
                       (batch_size, num_ref, img_dim)
        """

        if self.eof:
            return None

        batch_query = []
        batch_class_id = []

        for i in range(batch_size):
            line = self.fp.readline()
            if not line:
                self.eof = True
                break
            row = line.strip().split('\t')
            if len(row) == 4:
                key, url, class_name, feature = row
            else:
                key, class_name, feature = row
            class_id = self.class_names_to_ids[class_name]
            feature = [float(x) for x in feature.strip().split(",")]
            batch_query.append(feature)
            batch_class_id.append(class_id)
        batch_query = np.array(batch_query)
        batch_class_id = np.array(batch_class_id)

        # get ref set
        batch_ref = []
        for i in range(batch_size):
            batch_ref.append([self.ref_data[int(batch_class_id[i])][j] for j in range(self.num_ref)])
        batch_ref = np.stack(batch_ref, axis=0)

        return batch_class_id, batch_query, batch_ref
