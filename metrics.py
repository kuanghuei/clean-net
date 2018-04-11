# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""Define distance measurement and accuracy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def similarity(x1, x2):
    """Cosine similarity
    
    Args:
        x1: a feature vector
        x2: a feature vector
    Returns:
        cosine similarity between x1 and x2
    """
    with tf.variable_scope("similarity"):
        return tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(x1,1),tf.nn.l2_normalize(x2,1)), axis=1)


def accuracy(vlabel, cos_sim, threshold=0.1, scope="accuracy"):
    """Average accuracy
    
    Args:
        vlabel: verification labels
        cos_sim: predicted cosine similarity
        threshold: threshold on cosine similarity
    Returns:
        accuracy
    """
    with tf.variable_scope(scope):
        return tf.reduce_mean(tf.cast(tf.equal(vlabel, tf.cast(tf.greater_equal(cos_sim, tf.constant(threshold)), tf.float32)), tf.float32))
