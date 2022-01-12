# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from defaults    import *

import utils

from sklearn.impute import KNNImputer

import warnings
from sklearn.exceptions   import ConvergenceWarning
warnings.filterwarnings('ignore', category = ConvergenceWarning)

import numpy as np

import torch

# ==============================================================================
# Defs =========================================================================
def evaluate(
    n_neighbors,
    dataloader_test,
    eval_acc
):

    for i_mr, missing_rate in enumerate(missing_rates_eval):
        for j_mt, missing_type in enumerate(missing_types_eval):
            
            if missing_type == 'seq':
                introduceMissingEval = utils.IntroduceMissingSeq(missing_rate)
            else:
                introduceMissingEval = utils.IntroduceMissing(missing_rate)
            
            for i, data in enumerate(dataloader_test, 0):
                x, l = data
                
                # Introduce missingness
                x_mis, mask = introduceMissingEval(x)
                # (n, seq_len, kpis)
                x_mis = x_mis.permute((0, 2, 1))
                mask  = mask.permute((0, 2, 1))
                
                x_mis_np, _ = introduceMissingEval.mask_to_numpy(x_mis, mask)
                
                out_np = x_mis_np.copy()
                
                for j, elmt in enumerate(x_mis_np):
                    knn = KNNImputer(
                        n_neighbors = n_neighbors
                    )
                    out_np[j, :, :] = knn.fit_transform(elmt)
                
                # Replace nans with zeros
                x_mis_np = np.nan_to_num(x_mis_np)
                
                x_mis    = torch.tensor(x_mis_np)
                out      = torch.tensor(out_np)
                
                x_mis = x_mis.permute((0, 2, 1))
                out   = out.permute((0, 2, 1))
                mask  = mask.permute((0, 2, 1))
                
                eval_acc.accumulate(x, x_mis, mask, out, l, -1, 2*i_mr + j_mt)
            
    eval_acc.log(-1)
