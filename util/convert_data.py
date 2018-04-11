# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""Convert image features to numpy array for training and validation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='val',
                        help='train|val|all|ref')
    parser.add_argument('--class_list', default='/media/data/kualee/Food101N_CleanNetData/classes.txt',
                        help='List of class name')
    parser.add_argument('--data_path', default='/media/data/kualee/Food101N_CleanNetData/food101n_5k_val.tsv',
                        help='Path to datasets')
    parser.add_argument('--output_dir', default='/media/data/kualee/Food101N_CleanNetData/Food-101N/',
                        help='Output dir')
    parser.add_argument('--num_ref', default=32, type=int,
                        help='Number of reference image embeddings for a class.')
    opt = parser.parse_args()
    print(opt)
    
    # class data
    class_names = []
    with open(opt.class_list) as fp:
        for line in fp:
            class_names.append(line.strip())

    # create output dir if not exist
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    dest_npy = os.path.join(opt.output_dir, '{}.npy'.format(opt.split))

    # process data
    if opt.split == 'train':
        convert_verified(opt.data_path, dest_npy, class_names)
    elif opt.split == 'val':
        convert_verified(opt.data_path, dest_npy, class_names)
    elif opt.split == 'all':
        convert_all(opt.data_path, dest_npy, class_names)
    elif opt.split == 'ref':
        convert_ref(opt.data_path, dest_npy, opt.num_ref, class_names)
    else:
        raise ValueError('Unknown split: {}'.format(opt.split))
    

def convert_verified(data_file, dest_npy, class_names):
    """Convert data with verification labelsEncode query image feature
    
    Args:
        data_file: path to a tsv, where the columns are
                   [sample key, class name, verification label, h-dimensional feature delimited by ','] or
                   [sample key, image url, class name, verification label, h-dimensional feature delimited by ','].
                   Invalid samples have -1 as verification labels.
        dest_npy: destination file to save numpy array
        class_names: list of all class names in order
    """
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    positive_count = 0
    invalid_count = 0
    data = []
    with open(data_file) as fp:
        for i, line in enumerate(fp):
            row = line.strip().split('\t')
            if len(row) == 5:
                key, url, class_name, vlabel, feature = row
            else:
                key, class_name, vlabel, feature = row
            if vlabel != '-1':
                positive_count += int(vlabel)
                class_id = class_names_to_ids[class_name]
                vlabel = float(vlabel)
                feature = [float(x) for x in feature.strip().split(",")]
                data.append(feature+[float(class_id)]+[vlabel])
                sys.stdout.write('\r>> load %d samples with verification label' % (i+1))
            else:
                invalid_count += 1
    sys.stdout.write('\n')
    sys.stdout.flush()
    print("found {} positive samples".format(positive_count))
    print("found {} invalid samples".format(invalid_count))
    data = np.array(data)
    np.save(dest_npy, data)


def convert_all(data_file, dest_npy, class_names):
    """Convert data without verification labels
    
    Args:
        data_file: path to a tsv, where the columns are
                   [sample key, class name, h-dimensional feature delimited by ','] or
                   [sample key, image url, class name, h-dimensional feature delimited by ','].
                   Invalid samples have -1 as verification labels.
        dest_npy: destination file to save numpy array.
        class_names: list of all class names in order.
    """
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    data = [[] for _ in range(len(class_names))]
    with open(data_file) as fp:
        for i, line in enumerate(fp):
            row = line.strip().split('\t')
            if len(row) == 4:
                key, url, class_name, feature_str = row
            else:
                key, class_name, feature_str = row
            feature = [float(x) for x in feature_str.strip().split(",")]
            class_id = class_names_to_ids[class_name]
            data[class_id].append(feature)
            sys.stdout.write('\r>> load %d samples without verification label' % (i+1))
        sys.stdout.write('\n')
        sys.stdout.flush()
    data = np.array(data)
    np.save(dest_npy, data)


def convert_ref(data_file, dest_npy, num_ref, class_names):
    """Convert reference set
    
    Args:
        data_file: path to a tsv, where the columns are
                   [class name, reference feature id, h-dimensional feature delimited by ','] or
                   [class name, image url, reference feature id, h-dimensional feature delimited by ',']
        dest_npy: destination file to save numpy array.
        class_names: list of all class names in order.
    """
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    data = [[[] for _ in range(num_ref)] for _ in range(len(class_names))]
    with open(data_file) as fp:
        for line in fp:
            class_name, ref_id, feature_str = line.strip().split('\t')
            feature = [float(x) for x in feature_str.strip().split(",")]
            class_id = class_names_to_ids[class_name]
            ref_id = int(ref_id)
            data[class_id][ref_id] = feature
    data = np.array(data)
    np.save(dest_npy, data)


if __name__ == '__main__':
    main()
