#!/usr/bin/env python3

# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

# http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf

import sys

sys.path.append('../../common/')
from defaults    import *
from wgain_      import train
from data_mobile import loadData, normData, foldData
from eval_       import EvalACC
import utils

sys.path.append('../../common/nets/')
from net_ae      import NetAEConvTrans
from net_disc    import NetDiscConvTrans

import numpy as np

import torch
import torch.utils.data

import argparse

# ==============================================================================
# Settings =====================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--out_folder',         default = './out_test')
parser.add_argument('--missing_type',       default = 'ran')
parser.add_argument('--gpu_id',             default = None,   type = int)
parser.add_argument('--missing_rate_min',   default = 0.125,  type = float)
parser.add_argument('--missing_rate_max',   default = 0.875,  type = float)
parser.add_argument('--fold',               default = 0,      type = int)
args = parser.parse_args()

out_folder          = args.out_folder
missing_type        = args.missing_type
gpu_id              = args.gpu_id
missing_rate_min    = args.missing_rate_min
missing_rate_max    = args.missing_rate_max
fold                = args.fold

lr_ae               = 0.0001
wd_ae               = 1e-05

lr_disc             = 0.0001
wd_disc             = 0

alpha               = 10
iter_disc           = 5
lambda_gp           = 10


# ==============================================================================
# Data =========================================================================
utils.makeFolders(out_folder)

values_np, labels_np = loadData()
values_np            = normData(values_np)

values_np_train, values_np_test, labels_np_train, labels_np_test = foldData(values_np, labels_np, fold)


# ==============================================================================
# Data loaders =================================================================
dataset_train = torch.utils.data.TensorDataset(
    torch.tensor(values_np_train, dtype = torch.float),
    torch.tensor(labels_np_train, dtype = torch.long)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size  = batch_size,
    shuffle     = True,
    pin_memory  = True,
    num_workers = 3
)

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
# Definitions ==================================================================
if missing_type == 'seq':
    introduceMissingTrain = utils.IntroduceMissingSeqRand(missing_rate_min, missing_rate_max)
else:
    introduceMissingTrain = utils.IntroduceMissingRand(missing_rate_min, missing_rate_max)
      
# ==============================================================================
# Instantiation ================================================================
net_ae   = NetAEConvTrans(values_np.shape[1] * 2, values_np.shape[1])
net_disc = NetDiscConvTrans(values_np.shape[1], values_np.shape[1], bn = False, sigm = False)
eval_acc = EvalACC(values_np.shape[1] * 2, out_folder, fold, epochs_gain_varm[0], eval_acc_every)

net_dict = {
    "net_ae":   net_ae,
    "net_disc": net_disc
}


# ==============================================================================
# Move to GPU ==================================================================
device = torch.device("cuda:%d" % utils.gpuAssign(gpu_id))

net_ae.to(device)
net_disc.to(device)

eval_acc.to(device)


# ==============================================================================
# Opts =========================================================================
opt_ae = torch.optim.Adam(
    net_ae.parameters(),
    lr           = lr_ae,
    weight_decay = wd_ae
)

opt_disc = torch.optim.Adam(
    net_disc.parameters(),
    lr           = lr_disc,
    weight_decay = wd_disc
)

opt_dict = {
    "opt_ae":   opt_ae,
    "opt_disc": opt_disc
}


# ==============================================================================
# Calls ========================================================================
train(
    alpha,
    iter_disc,
    lambda_gp,
    introduceMissingTrain,
    net_dict,
    opt_dict,
    dataloader_train,
    dataloader_test,
    device,
    eval_every,
    out_folder,
    eval_acc,
    epochs_end   = epochs_gain_varm[1],
    epochs_start = epochs_gain_varm[0]
)
