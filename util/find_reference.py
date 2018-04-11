# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""Find reference features through K-means clustering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_list', default='/media/data/kualee/Food101N_CleanNetData/classes.txt',
                        help='list of class name')
    parser.add_argument('--input_npy', default='/media/data/kualee/Food101N_CleanNetData/Food-101N/all.npy',
                        help='path npy of all image features')
    parser.add_argument('--output_dir', default='/media/data/kualee/Food101N_CleanNetData/Food-101N/',
                        help='output dir')
    parser.add_argument('--num_ref', default=32, type=int,
                        help='Number of reference image embeddings, i.e. number of clusters, for a class.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Image feature dimension.')
    opt = parser.parse_args()

    class_names = []
    with open(opt.class_list) as fp:
        for line in fp:
            class_names.append(line.strip())
    
    dest_npy = os.path.join(opt.output_dir, 'ref.npy')

    print("Finding reference features...")
    convert_ref(opt.input_npy, dest_npy, opt.num_ref, opt.img_dim, class_names)


def convert_ref(input_npy, dest_npy, num_ref, img_dim, class_names):
    """convert data 
    Args:
        data_file: a tsv where each columns are: key, class_name, h-dimensional feature
        dest_npy: the destination npy to save
                  dim - (num_class, ref_id, img_dim)
                  img_id - order of a sample in the data_file
        num_ref: number of reference feature per class
    """
    print("Loading all image features...")
    input_data = np.load(input_npy) #(num_class, num_class_sample, img_dim)
    output_data = np.zeros((len(class_names), num_ref, img_dim))

    for class_id, class_name in enumerate(class_names):
        class_features = input_data[class_id]
        print("Convert to numpy " + class_name)
        Z = np.array(class_features, dtype=np.float32)
        print("Starting K-means " + class_name)
        output_data[class_id,:,:] = K_means(Z, num_ref)
    print('output_data', output_data.shape)
    np.save(dest_npy, output_data)


def K_means(Z, num_ref):
    """run k-means
    Args:
        Z: image features within a class
        num_ref: number of reference embeddings
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
    ret,label,center=cv2.kmeans(Z,num_ref,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    return center


if __name__ == '__main__':
    main()
