"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet
import math
from sklearn.model_selection import KFold

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff))
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2))
        norm = np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
        similarity = dot / norm
        # dist = np.arccos(np.minimum(1, similarity)) / math.pi
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def find_unvalid(image_paths, embeddings1, embeddings2, actual_issame, distance_metric, subtract_mean=False, nrof_folds=10):
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    indices = np.arange(nrof_pairs)
    optimum = 0.38

    base = 300
    i = 0

    same_dist = []
    diff_dist = []

    f_file = open(os.path.join("E:\scrawl_images\star_images_160", 'f.txt'), 'a', encoding='utf-8')
    for (i1, i2) in zip(embeddings1, embeddings2):
        if subtract_mean:
            mean = np.mean(np.concatenate([i1, i2]))
        else:
            mean = 0.0
        dist = distance(i1 - mean, i2 - mean, distance_metric)
        if (int((i / base)) % 2) == 0:
            same_dist.append(dist)
            if dist > 0.38:
                print("第 %d 个相同图片计算距离为: %f, 图片为: %s %s" % (i, dist, image_paths[i*2], image_paths[i*2 + 1]))
        else:
            diff_dist.append(dist)
            if dist < 0.3:
                print("第 %d 个不同图片计算距离为: %f, 图片为: %s %s" % (i, dist, image_paths[i*2], image_paths[i*2 + 1]))
                f_file.write(image_paths[i*2] + "\t" + image_paths[i*2 + 1] + "\t" + str(dist) + "\n")
        print("i: %d, dist: %f" % (i, dist))
        i+=1
    print("同一个人距离平均: %f, 最大: %f" % (np.mean(same_dist), np.max(same_dist)))
    print("不同人距离平均: %f, 最小: %f" % (np.mean(diff_dist), np.min(diff_dist)))

    '''
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0

        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)
        '''

def evaluate(image_paths, embeddings, actual_issame, nrof_folds, distance_metric, subtract_mean):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    # 从第0个开始，步进2取元素
    embeddings1 = embeddings[0::2]
    # 从第1个开始，步进2取元素
    embeddings2 = embeddings[1::2]
    find_unvalid(image_paths, embeddings1, embeddings2, actual_issame, distance_metric, subtract_mean)

    '''
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
                                               np.asarray(actual_issame), nrof_folds=nrof_folds,
                                               distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
                                              np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds,
                                              distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far
    '''


def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r', encoding='UTF-8') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)



