# ---------------------------------------------------------------
# CleanNet implementation based on https://arxiv.org/abs/1711.07131.
# "CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise"
# Kuang-Huei Lee, Xiaodong He, Lei Zhang, Linjun Yang
#
# Writen by Kuang-Huei Lee, 2018
# Licensed under the MSR-LA Full Rights License [see license.txt]
# ---------------------------------------------------------------
"""Heldout certain training data for certain classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--heldout_class_list', default='/media/data/kualee/Food101N_CleanNetData/classes_heldout_50.tsv',
                        help='List of heldout ')
    parser.add_argument('--input_file', default='/media/data/kualee/Food101N_CleanNetData/food101n_55k_train.tsv',
                        help='Input file')
    parser.add_argument('--output_file', default='/media/data/kualee/Food101N_CleanNetData/food101n_55k_train_heldout50.tsv',
                        help='Output file')
    opt = parser.parse_args()
    print(opt)

    classes_heldout = set()
    with open(opt.heldout_class_list) as fp:
    	for line in fp:
	    class_name, keep = line.strip().split('\t')
	    if keep == '0':
		classes_heldout.add(class_name)

    with open(opt.input_file) as fin, open(opt.output_file, 'w') as fout:
        for i, line in enumerate(fin):
            row = line.strip().split('\t')
            if len(row) == 5:
                key, url, class_name, vlabel, feature = row
            else:
                key, class_name, vlabel, feature = row
            if class_name not in classes_heldout:
                fout.write(line)
