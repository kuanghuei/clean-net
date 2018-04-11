# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""Data provider factory"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math
import os
import sys

import numpy as np
import tensorflow as tf

import data_provider_trainval
import data_provider_inference


datasets_map = {
    'trainval': data_provider_trainval,
    'inference': data_provider_inference,
}


def get_data_batcher(name, mode, opt):
    """Given a dataset name and data provider mode returns a Dataset.

    Args:
        name: String, the name of the dataset.
        mode: train/val/unverified/inference
        opt: Options

    Returns:
        a data batcher

    Raises:
        ValueError: If the dataset `name` is unknown.
    """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_data_batcher(mode, opt)

