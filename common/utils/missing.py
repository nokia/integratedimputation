# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import sys

sys.path.append('../')
from defaults    import *

import torch
import random
import numpy as np

# ==============================================================================
# Function definitions =========================================================
class IntroduceMissing():
    def __init__(self, missing_rate):
        super().__init__()
        
        self.missing_rate = missing_rate

    def __call__(self, x):
        with torch.no_grad():
            if self.missing_rate != 0:
                mask = torch.rand_like(x)
                mask = mask / self.missing_rate
                mask = torch.clamp(mask, 0, 1)
                mask = torch.trunc(mask)
            else:
                mask = torch.ones_like(x)
        
        x_mis = x * mask
        
        return(x_mis, mask)
        
    def mask_to_numpy(self, x_mis, mask):
        x_mis_np = x_mis.numpy()
        mask_np  = mask.numpy()
        
        x_mis_np[mask_np==0] = np.nan
        
        return(x_mis_np, mask_np)
        
class IntroduceMissingSeq(IntroduceMissing):
    def __init__(self, missing_rate):
        super().__init__(missing_rate)

    def __call__(self, x):
        with torch.no_grad():
            n_kpis       = x.shape[-2]
                
            missing_rate = round(self.missing_rate / (missing_seq_len/seq_len)) * (missing_seq_len/seq_len)
                
            n_seq_mis    = int(seq_len * missing_rate / missing_seq_len)
                
            if n_seq_mis != 0:
                #print('random value:', self.missing_rate, '--> calculated:', missing_rate, f'({n_seq_mis} sequences)')

                max_val      = seq_len - (missing_seq_len-1) - missing_seq_len * (n_seq_mis-1)
                
                i_seq_start  = [[missing_seq_len*i + x for i, x in enumerate(sorted(np.random.choice(max_val, n_seq_mis)))]
                                 for j in range(x.shape[0])]

                i_seq_start  = torch.tensor(i_seq_start, device=x.device).unsqueeze(-1)
                
                i_seq_steps  = torch.arange(missing_seq_len, device=x.device).unsqueeze(0).unsqueeze(0)
                
                i_seq = i_seq_start + i_seq_steps

                i_seq = i_seq.view(-1, n_seq_mis*missing_seq_len)
                
                # unsqueeze for all kpis 
                i_seq = i_seq.unsqueeze(1).repeat(1, n_kpis, 1)

                ones  = torch.ones_like(x)

                # scatter zeros at indices
                mask = ones.scatter_(2, i_seq, 0)

            else:
                mask = torch.ones_like(x)

        x_mis = x * mask

        return(x_mis, mask)
        
class IntroduceMissingRand(IntroduceMissing):
    def __init__(self, missing_rate_min, missing_rate_max):
        super().__init__(None)
        
        self.missing_rate_min = missing_rate_min
        self.missing_rate_max = missing_rate_max
        

    def __call__(self, x):
        
        self.missing_rate = random.uniform(self.missing_rate_min, self.missing_rate_max)
        
        return(super().__call__(x))
        
class IntroduceMissingSeqRand(IntroduceMissingSeq):
    def __init__(self, missing_rate_min, missing_rate_max):
        super().__init__(None)
        
        self.missing_rate_min = missing_rate_min
        self.missing_rate_max = missing_rate_max
        

    def __call__(self, x):
        
        self.missing_rate = random.uniform(self.missing_rate_min, self.missing_rate_max)
        
        return(super().__call__(x))
