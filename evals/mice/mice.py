#!/usr/bin/env python3

# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import sys

sys.path.append('../../common/')
from defaults    import *
from mice_ import evaluate
from data_mobile import loadData, normData, foldData
from eval_       import EvalACC
import utils

import numpy as np

import torch
import torch.utils.data

import argparse

# ==============================================================================
# Settings =====================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--out_folder',     default = './out_test')
parser.add_argument('--gpu_id',         default = None, type = int)
parser.add_argument('--fold',           default = 0,    type = int)
args = parser.parse_args()

out_folder   = args.out_folder
gpu_id       = args.gpu_id
fold         = args.fold

iterations   = 1
n_estimators = 10

# ==============================================================================
# Data =========================================================================
utils.makeFolders(out_folder)

values_np, labels_np = loadData()
values_np            = normData(values_np)

values_np_train, values_np_test, labels_np_train, labels_np_test = foldData(values_np, labels_np, fold)

dataset_test = torch.utils.data.TensorDataset(
    torch.tensor(values_np_test, dtype = torch.float),
    torch.tensor(labels_np_test, dtype = torch.long)
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size  = batch_size,
    shuffle     = False,
    pin_memory  = True,
    num_workers = 3
)

# ==============================================================================
# Instantiation ================================================================
eval_acc = EvalACC(values_np.shape[1] * 2, out_folder, fold)

# ==============================================================================
# Move to GPU ==================================================================
device = torch.device("cuda:%d" % utils.gpuAssign(gpu_id))

eval_acc.to(device)

# ==============================================================================
# Calls ========================================================================
evaluate(
    iterations,
    dataloader_test,
    eval_acc
)
