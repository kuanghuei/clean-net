# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""CleanNet model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class CleanNet(object):
    """CleanNet model"""
    def __init__(self, num_ref, img_dim, embed_norm, dropout_rate=0.0, weight_decay=0.0):
        """Basic setup

        Args:
            num_ref: number of reference embeddings
            img_dim: dimension of image feature
            embed_norm: type of embedding normalization to use
            dropout_rate: drop-out rate
            weight_decay: l2 regularization
        """
        self.num_ref = num_ref # n_cluster
        self.img_dim = img_dim #n_feature
        self.embed_norm = embed_norm
        self.keep_prob = 1.0 - dropout_rate
        self.weight_decay = weight_decay

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.reference = tf.placeholder(tf.float32, shape=(None, self.num_ref, self.img_dim), name="reference")
        self.query = tf.placeholder(tf.float32, shape=(None, self.img_dim), name="query")


    def forward(self, is_training):
        """CleanNet forward pass

        Args:
            is_training: True - training model / False - inference model
        Returns:
            phi_s: reference set vector
            v_q: query image feature (batch_size, img_dim)
            phi_q: query vector
            v_qr: reconstructed v_q
        """

        # normalization
        if self.embed_norm == 'log':
            v_q = tf.log(self.query + 1.0)
            v_S = tf.log(self.reference + 1.0)
        elif self.embed_norm == 'l2norm':
            v_q = tf.nn.l2_normalize(self.query, 1)
            v_S = tf.nn.l2_normalize(self.reference, 2)
        elif self.embed_norm == 'no_norm':
            v_q = self.query
            v_S = self.reference
        else:
            raise NotImplementedError

        # encode query
        phi_q, v_qr = self.query_encoder(v_q, is_training=is_training)
        # encode reference set
        phi_s = self.ref_set_encoder(v_S, is_training=is_training)
        
        return phi_s, v_q, phi_q, v_qr


    def query_encoder(self, v_q, is_training=True, scope="query_encoder"):
        """Encode query image feature
        
        Args:
            v_q: query image feature (batch_size, img_dim)
            is_training: True - training model / False - inference model
        Returns:
            phi_q: query vector
            v_qr: reconstructed v_q
        """
        with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(inputs=v_q, 
                num_outputs=256,
                activation_fn=tf.nn.tanh,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                biases_initializer=tf.zeros_initializer())
            phi_q = tf.contrib.layers.fully_connected(inputs=h1, 
                num_outputs=128,
                activation_fn=tf.nn.tanh,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                biases_initializer=tf.zeros_initializer())
            h2 = tf.contrib.layers.fully_connected(inputs=phi_q, 
                num_outputs=256,
                activation_fn=tf.nn.tanh,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                biases_initializer=tf.zeros_initializer())
            v_qr = tf.contrib.layers.fully_connected(inputs=h2, 
                num_outputs=self.img_dim,
                activation_fn=tf.nn.tanh,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                biases_initializer=tf.zeros_initializer())
            return phi_q, v_qr
    

    def ref_set_encoder(self, v_S, is_training=True, scope="ref_set_encoder"):
        """Encode reference image features
        
        Args:
            v_S: query image feature (batch_size, img_dim)
            is_training: True - training model / False - inference model
        Returns:
            phi_s: reference set vector
        """
        with tf.variable_scope(scope) as scope:
            context = tf.get_variable(name='context',
                                      shape=[256],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32)

            h0 = tf.contrib.layers.fully_connected(inputs=v_S, 
                num_outputs=512,
                activation_fn=tf.nn.tanh,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                biases_initializer=tf.zeros_initializer())

            if is_training:
                h = tf.nn.dropout(h0, keep_prob=self.keep_prob)
            
            h = tf.contrib.layers.fully_connected(inputs=h0, 
                num_outputs=256,
                activation_fn=tf.nn.tanh,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                biases_initializer=tf.zeros_initializer())
            
            u = tf.contrib.layers.fully_connected(inputs=h,
                num_outputs=256,
                activation_fn=tf.nn.tanh,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                biases_initializer=tf.zeros_initializer())

            attn = tf.reduce_sum(tf.multiply(u, context), axis=2, keepdims=True)
            alpha = tf.nn.softmax(attn, axis=1)
            attended_set_vector = tf.reduce_sum(tf.multiply(h, alpha), axis=1)

            phi_s = tf.contrib.layers.fully_connected(inputs=attended_set_vector, 
                num_outputs=128,
                activation_fn=tf.nn.tanh,
                weights_regularizer= tf.contrib.layers.l2_regularizer(self.weight_decay),
                biases_initializer=tf.zeros_initializer())

            return phi_s
