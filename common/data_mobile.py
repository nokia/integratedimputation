# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from defaults import seed, folds, k_mob, seq_len, overlap, eps

import numpy as np


# ==============================================================================
# Settings =====================================================================
in_folder    = '../../data'
cols_to_keep = [5, 6, 7, 10, 14, 17, 18, 19, 20, 21, 22, -1]

# ==============================================================================
# Data handling ================================================================
def printCols():
    cols_np = np.load(f'{in_folder}/kpi_names.npy', allow_pickle = True)

    cols_np = cols_np[cols_to_keep]
    for i, col in enumerate(cols_np):
        print(i, col, sep = ": ")

def loadData():
    data_np = np.load(f'{in_folder}/kpi_data.npy', allow_pickle = True)
    
    values_np = data_np[:, :, cols_to_keep]
    labels_np = values_np[:, 0, -1].astype(np.int)
    values_np = np.delete(values_np, -1, 2)

    # Permute to NCL
    values_np = np.transpose(values_np, (0, 2, 1))

    return(values_np, labels_np)
    

def normData(values_np):
    # Normalize channels individually
    values_mean = np.mean(values_np, (0, 2), keepdims = True)
    values_std  = np.std(values_np,  (0, 2), keepdims = True)

    values_np   = (values_np - values_mean) / (values_std + eps)
    
    return(values_np)
    
def splitSequences(data_np):
    start_ids_np = np.arange(data_np.shape[-1] - seq_len, step = seq_len - overlap)
    start_ids_np = np.expand_dims(start_ids_np, 1)

    step_ids_np = np.arange(seq_len)
    step_ids_np = np.expand_dims(step_ids_np, 0)

    seq_ids_np = start_ids_np + step_ids_np
    
    seq_np = data_np[..., seq_ids_np]
    seq_np = np.transpose(seq_np, (0, 2, 1, 3))

    return(seq_np)
 
# WARNING ======================================================================   
# The whole function is built on the assumption that group size is divisible by
# folds!
# ==============================================================================
def foldData(values_np, labels_np, fold):

    # Train/test split
    size       = values_np.shape[0]
    group_size = size // k_mob
    fold_size  = group_size // folds

    rnd        = np.random.RandomState(seed)
    perm_list  = np.expand_dims([rnd.permutation(group_size) for i in range(k_mob)], 0)
    perm_np    = np.concatenate(perm_list, 0)
    perm_np    = perm_np + np.expand_dims(np.arange(size, step = group_size), 1)
    
    test_ids = perm_np[:, (fold * fold_size):((fold + 1) * fold_size)].flatten()
    
    values_test_np  = values_np[test_ids, ...]
    values_train_np = np.delete(values_np, test_ids, 0)
    
    labels_test_np  = labels_np[test_ids]
    labels_train_np = np.delete(labels_np, test_ids, 0)

    # Sequence split
    values_train_np = splitSequences(values_train_np)
    values_test_np  = splitSequences(values_test_np)

    labels_train_np = np.repeat(np.expand_dims(labels_train_np, 1), values_train_np.shape[1], axis = 1)
    labels_test_np  = np.repeat(np.expand_dims(labels_test_np, 1), values_test_np.shape[1], axis = 1)
    
    # Merge users with seqeunces
    values_train_np = np.reshape(values_train_np, (-1, values_train_np.shape[2], values_train_np.shape[3]))
    values_test_np  = np.reshape(values_test_np, (-1, values_test_np.shape[2], values_test_np.shape[3]))
    
    labels_train_np = labels_train_np.flatten()
    labels_test_np  = labels_test_np.flatten()

    return(values_train_np, values_test_np, labels_train_np, labels_test_np)
