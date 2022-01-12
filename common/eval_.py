# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import torch

from defaults       import *
from nets.net_class import NetClass

import utils

import glob
import os
import itertools

# ==============================================================================
# Function definitions =========================================================
class EvalACC(utils.DmModule):
    def __init__(self, w_in, out_folder, fold, epochs_start = 0, eval_every = 1, classif_path = '../classif'):
        super().__init__()
        
        self.eval_every = eval_every
        
        self.mseLogger  = utils.Logger([f"mse_{m_type}_" + str(m_rate).replace(".", "") for m_rate, m_type in list(itertools.product(missing_rates_eval, missing_types_eval))], epoch_digits, out_folder, epochs_start != 0, log_name = 'log_mse.txt')
        self.accLogger  = utils.Logger([f"acc_{m_type}_" + str(m_rate).replace(".", "") for m_rate, m_type in list(itertools.product(missing_rates_eval, missing_types_eval))], epoch_digits, out_folder, epochs_start != 0, log_name = 'log_acc.txt')
        
        # Load net
        self.net_class  = NetClass(w_in, k_mob)
        model_names     = glob.glob(f'{classif_path}/out_fixm_00_ran_{fold}/net_class*')
        model_names.sort()        
        model_name      = os.path.basename(model_names[-1]).replace('.pth', '')
        folder          = os.path.dirname(model_names[-1])

        utils.loadNet(self.net_class, model_name = model_name, folder = folder)
        
    def accumulate(self, x, x_mis, mask, out, labs, epoch, position):
        if not epoch % self.eval_every:
            self.net_class.eval()
            with torch.no_grad():
                x     = x.to(self.device)
                x_mis = x_mis.to(self.device)
                mask  = mask.to(self.device)
                out   = out.to(self.device)
                labs  = labs.to(self.device)
                
                # MSE
                out_imp = x_mis + ((1 - mask) * out)
                l_mse = torch.nn.MSELoss()(out_imp, x)
                
                self.mseLogger.accumulate([l_mse.item()], [position])
                
                # ACC
                mask_in  = torch.full_like(out_imp, 1.0) # All data is "present"
                x_mis_in = torch.cat([out_imp, mask_in], dim = 1)

                ass = torch.argmax(self.net_class(x_mis_in), 1, keepdim = False)
                acc = (labs == ass).sum().item() / ass.shape[0]
                
                self.accLogger.accumulate([acc], [position])

    def log(self, epoch):
        if not epoch % self.eval_every:
            self.mseLogger.log(epoch)
            self.accLogger.log(epoch)