# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""CleanNet training script"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil

import tensorflow as tf
import numpy as np

import data_provider_factory
from model import CleanNet
from metrics import similarity, accuracy
from losses import total_loss


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data/',
                           'Data directory containing {train|val|all|ref}.npy.')
tf.app.flags.DEFINE_string('checkpoint_dir', './train/checkpoints/',
                           'Directory to keep checkpoints.')
tf.app.flags.DEFINE_string('log_dir', './train/log/',
                           'Directory to keep summaries and logs.')
tf.app.flags.DEFINE_integer('log_interval', 100,
                            'Step interval to print training log.')
tf.app.flags.DEFINE_integer('val_interval', 2000,
                            'Step interval to run validation and save models.')
tf.app.flags.DEFINE_integer('img_dim', 2048,
                            'Dimensionality of the image embedding')
tf.app.flags.DEFINE_integer('num_ref', 32,
                            'Number of reference image embeddings for a class')
tf.app.flags.DEFINE_integer('batch_size_sup', 32,
                            'Number of samples with human supervision in a training mini-batch.')
tf.app.flags.DEFINE_integer('batch_size_unsup', 32,
                            'Number of samples without human supervision in a training mini-batch.')
tf.app.flags.DEFINE_integer('val_batch_size', 64,
                            'Validation batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          'Initial learning rate.')
tf.app.flags.DEFINE_integer('n_step', 60000,
                            'Number of steps to train.')
tf.app.flags.DEFINE_integer('lr_update', 30000,
                            'Number of steps to update the learning rate.')
tf.app.flags.DEFINE_float('lr_decay', 0.1,
                          'Learning rate decay factor.')
tf.app.flags.DEFINE_float('dropout_rate', 0.0,
                          'Dropout rate.')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 
                          'Weight decay, for regularization.')
tf.app.flags.DEFINE_float('neg_weight', 5.0, 
                          'Negative sample weight.')
tf.app.flags.DEFINE_float('val_sim_thres', 0.1, 
                          'Similarity threshold for validation.')
tf.app.flags.DEFINE_float('momentum', 0.9, 
                          'SGD momentum.')
tf.app.flags.DEFINE_string('embed_norm', 'log',
                           'log|l2_norm|no_norm')

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    tf.logging.info("Start training...")
    train()


def train():
    """Training"""
    opt = FLAGS
    
    tf.logging.info("Build CleanNet...")
    batch_size = opt.batch_size_sup + opt.batch_size_unsup
    model = CleanNet(opt.num_ref, opt.img_dim, opt.embed_norm, opt.dropout_rate, opt.weight_decay)

    # phi_s: class embedding (batch_size, embed_size)
    # v_q: query image feature (batch_size, img_dim)
    # phi_q: query embedding (batch_size, embed_size)
    # v_qr: reconstructed query image feature (batch_size, img_dim)
    phi_s, v_q, phi_q, v_qr = model.forward(is_training=True)
    
    # verification labels
    vlabel = tf.placeholder(tf.float32, shape=(None,), name="vlabel")
    
    # verification flags indicating a sample is for supervised(1) or unsupervised(0) training
    vflag = tf.placeholder(tf.float32, shape=(None,), name="vflag")
    
    cos_sim = similarity(phi_s, phi_q)

    acc = accuracy(vlabel[:opt.batch_size_sup], cos_sim[:opt.batch_size_sup], threshold=0.1, scope="train_acc")
    val_acc = accuracy(vlabel, cos_sim, threshold=opt.val_sim_thres, scope="val_acc_at_{}".format(opt.val_sim_thres))
    tf.summary.scalar('train/accuracy', acc)
    
    objective_loss = tf.reduce_mean(total_loss(vlabel, cos_sim, phi_s, v_q, phi_q, v_qr, vflag, opt.neg_weight, beta=0.1, gamma=0.1))
    tf.summary.scalar('train/objective_loss', objective_loss)
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('train/regularization_loss', regularization_loss)
    loss = objective_loss + regularization_loss
    tf.summary.scalar('train/loss', loss)

    lr = tf.train.exponential_decay(opt.learning_rate, model.global_step, opt.lr_update, opt.lr_decay, staircase=True)
    tf.summary.scalar('train/lr', lr)
    merged = tf.summary.merge_all()

    optimizer = tf.train.MomentumOptimizer(lr, opt.momentum)
    train_op = optimizer.minimize(loss, global_step=model.global_step)

    tf.logging.info("Get data batcher...")
    supervised_data = data_provider_factory.get_data_batcher('trainval', 'train', opt)
    val_data = data_provider_factory.get_data_batcher('trainval', 'val', opt)
    if opt.batch_size_unsup > 0:
        unsupervised_data = data_provider_factory.get_data_batcher('trainval', 'unverified', opt)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        train_summary_writer = tf.summary.FileWriter(opt.log_dir + '/train', sess.graph)
        val_summary_writer = tf.summary.FileWriter(opt.log_dir + '/val')

        cur_step = 0
        best_avg_val_acc = 0.0
        sess.run(init_op)

        # recover from latest checkpoint and run validation if available
        ckpt = tf.train.get_checkpoint_state(opt.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            cur_step, avg_val_acc = validation(sess, model, loss, val_acc, vlabel, vflag, opt.val_batch_size, val_data, val_summary_writer)
            best_avg_val_acc = avg_val_acc
            tf.logging.info("Recover model at global step = %d.", cur_step)
        else:
            tf.logging.info("Training from scratch.")

        while cur_step < opt.n_step:
            # data for supervised training
            _, batch_vlabel, batch_q, batch_vflag, batch_ref = supervised_data.get_batch(batch_size=opt.batch_size_sup)

            # data for unsupervised training
            if opt.batch_size_unsup > 0:
                # ubatch_vlabel_u is a dummy zero tensor since unsupervised samples don't have verification labels
                _, ubatch_vlabel_u, ubatch_q, ubatch_vflag, ubatch_ref = unsupervised_data.get_batch(batch_size=opt.batch_size_unsup)

                # concate supervised and unsupervied training data
                batch_vlabel = np.concatenate([batch_vlabel, ubatch_vlabel_u], axis=0)
                batch_q = np.concatenate([batch_q, ubatch_q], axis=0)
                batch_vflag = np.concatenate([batch_vflag, ubatch_vflag], axis=0)
                batch_ref = np.concatenate([batch_ref, ubatch_ref], axis=0)

            _, cur_step, cur_loss, cur_acc, summary = sess.run([train_op, model.global_step, loss, acc, merged], 
                   feed_dict={model.reference: batch_ref, 
                              model.query: batch_q, 
                              vlabel: batch_vlabel,
                              vflag: batch_vflag})

            train_summary_writer.add_summary(summary, cur_step)

            if cur_step % opt.log_interval == 0:
                tf.logging.info('step {}: train/loss = {}, train/acc = {}'.format(cur_step, cur_loss, cur_acc))
            if cur_step % opt.val_interval == 0 and cur_step != 0:
                _, avg_val_acc = validation(sess, model, loss, val_acc, vlabel, vflag, opt.val_batch_size, val_data, val_summary_writer)
                if not os.path.exists(opt.checkpoint_dir):
                    os.mkdir(opt.checkpoint_dir)
                save_path = saver.save(sess, opt.checkpoint_dir)
                print("Model saved in path: %s" % save_path)
                if avg_val_acc > best_avg_val_acc:
                    best_avg_val_acc = avg_val_acc
                    model_path = os.path.join(save_path, "checkpoint")
                    best_model_path = os.path.join(save_path, "best_model_{}".format(cur_step))
                    shutil.copy(model_path, best_model_path)
                    print("Best model saved in path: %s" % best_model_path)


def validation(sess, model, loss, val_acc, vlabel, vflag, batch_size, data, summary_writer):
    """Run validation"""
    cumulative_loss = 0.
    cumulative_acc = 0.
    data.reset()
    steps = int(data.data_size/batch_size)
    for i in range(steps):
        _, batch_vlabel, batch_q, batch_vflag, batch_ref = data.get_batch(batch_size)
        cur_step, cur_loss, cur_val_acc = sess.run([model.global_step, loss, val_acc], 
            feed_dict={model.reference: batch_ref, model.query: batch_q, vlabel: batch_vlabel, vflag: batch_vflag})
        cumulative_loss += cur_loss
        cumulative_acc += cur_val_acc
        sys.stdout.write('\r>> Evaluated losses for %d of %d batches.' % (i+1, steps))
    sys.stdout.write('\n')
    sys.stdout.flush()

    avg_loss = cumulative_loss/steps
    avg_val_acc = cumulative_acc/steps
    summary = tf.Summary(value=[tf.Summary.Value(tag="val/loss", simple_value=avg_loss), tf.Summary.Value(tag="val/acc", simple_value=avg_val_acc)])
    summary_writer.add_summary(summary, cur_step)
    summary_writer.flush()
    tf.logging.info('step {}: val/loss = {}, val/acc = {}'.format(cur_step, avg_loss, avg_val_acc))
    return cur_step, avg_val_acc


if __name__ == '__main__':
    tf.app.run()
