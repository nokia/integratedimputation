# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from defaults    import *

import utils

import numpy as np

import torch

# ==============================================================================
# Defs =========================================================================
def evaluate(
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
                
                sum_vals = torch.sum(x_mis, dim = 2)
                num_elem = torch.sum(mask,  dim = 2) + eps
                avg_vals = sum_vals / num_elem
                avg_vals = avg_vals.unsqueeze(2)
                
                out = x_mis + ((1 - mask) * avg_vals)
                
                eval_acc.accumulate(x, x_mis, mask, out, l, -1, 2*i_mr + j_mt)
            
    eval_acc.log(-1)
