# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

# General ----------------------------------------------------------------------
seed         = 982347

folds        = 5
fold_digits  = 1

run_digits   = 2
epoch_digits = 4

# Mobile data ------------------------------------------------------------------
k_mob        = 8
seq_len      = 256
overlap      = 128

eps          = 1e-08

# Missing ----------------------------------------------------------------------
missing_seq_len     = 8
missing_rates_train = [0.0, 0.25, 0.5, 0.75]
missing_types_eval  = ['ran', 'seq']
missing_rates_eval  = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

# Training ---------------------------------------------------------------------
batch_size     = 64

eval_every     = 5
eval_acc_every = 50

epochs_classif_fixm = [0, 300]
epochs_classif_varm = [0, 500]

epochs_gain_fixm    = [0, 500]
epochs_gain_varm    = [0, 750]

epochs_mida_fixm    = [0, 300]
epochs_mida_varm    = [0, 500]
