# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""Define losses"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def reconst_loss(x, r_x):
    """autoencoder reconstruction loss
    
    Args:
        x: feature vector
        r_x: reconstructed feature vector
    Returns:
        reconstruction loss
    """
    with tf.variable_scope("reconst_loss"):
        return tf.reduce_sum(tf.square(tf.subtract(x, r_x)), axis=1)


def cos_sim_loss(vlabel, cos_sim, neg_weight, margin=0.1):
    """Cosine similarity loss
    
    Args:
        vlabel: verification label
        cos_sim: predicted cosine similarity
        neg_weight: negative sample weight
        margin: cosine similarity margin
    Returns:
        cosine similarity loss
    """
    with tf.variable_scope("cos_sim_loss"):
        return tf.add(tf.multiply(vlabel, 1.0 - cos_sim), \
            neg_weight * (tf.multiply(1.0 - vlabel, tf.maximum(tf.constant(0.0), cos_sim - margin))))


def unsup_cos_sim_loss(cos_sim, neg_weight, margin=0.1):
    """Unsupervised cosine similarity loss
    
    Args:
        cos_sim: predicted cosine similarity
        neg_weight: negative sample weight
        margin: cosine similarity margin
    Returns:
        unsupervised cosine similarity loss
    """
    with tf.variable_scope("unsup_cos_sim_loss"):
        sudo_vlabel = tf.cast(tf.greater_equal(cos_sim, 0.1), dtype=tf.float32)
        return cos_sim_loss(sudo_vlabel, cos_sim, neg_weight)


def total_loss(vlabel, cos_sim, phi_s, v_q, phi_q, v_qr, vflag, neg_weight, beta=0.1, gamma=0.1):
    """Total loss
    
    Args:
        vlabel: verification label
        cos_sim: predicted cosine similarity
        phi_s: reference set vector
        v_q: query image feature
        phi_q: query vector
        v_qr: reconstructed v_q
        vflag: verification flags indicating a sample is for supervised(1) or unsupervised(0) training
        neg_weight: negative sample weight
        beta: weight on reconstruction loss
        gamma: weight on unsupervised cosine similarity loss
    Returns:
        supervised cosine similarity loss + unsupervised cosine similarity loss + reconstruction loss
    """

    with tf.variable_scope("total_loss"):
        return tf.multiply(cos_sim_loss(vlabel, cos_sim, neg_weight), vflag) + \
               beta *reconst_loss(v_q, v_qr) + \
               gamma * tf.multiply(unsup_cos_sim_loss(cos_sim, neg_weight), 1.0-vflag)
