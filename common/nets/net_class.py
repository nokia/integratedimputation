# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import sys

sys.path.append('../common')
from defaults import * 

import utils

import torch

# ==============================================================================
# Settings =====================================================================
n_conv_1 = 64
n_conv_2 = 128
n_conv_3 = 256
n_fcon_4 = 128


# ==============================================================================
# Module definitions ===========================================================
class NetClass(utils.DmModule):
    def __init__(self, w_in, w_out):
        super().__init__()
        
        self.w_out  = w_out

        self.conv_1 = torch.nn.Conv1d(w_in, n_conv_1, 3, stride = 1, padding = 1)
        self.norm_1 = torch.nn.BatchNorm1d(n_conv_1)
        self.nonl_1 = torch.nn.LeakyReLU()

        self.pool_1 = torch.nn.MaxPool1d(4, stride = 4)

        self.conv_2 = torch.nn.Conv1d(n_conv_1, n_conv_2, 3, stride = 1, padding = 1)
        self.norm_2 = torch.nn.BatchNorm1d(n_conv_2)
        self.nonl_2 = torch.nn.LeakyReLU()

        self.pool_2 = torch.nn.MaxPool1d(4, stride = 4)

        self.conv_3 = torch.nn.Conv1d(n_conv_2, n_conv_3, 3, stride = 1, padding = 1)
        self.norm_3 = torch.nn.BatchNorm1d(n_conv_3)
        self.nonl_3 = torch.nn.LeakyReLU()
        
        self.pool_3 = torch.nn.MaxPool1d(4, stride = 4)

        self.fcon_4 = torch.nn.Linear(
            in_features  = n_conv_3 * (seq_len // 64),
            out_features = n_fcon_4
        )
        self.norm_4 = torch.nn.BatchNorm1d(n_fcon_4)
        self.nonl_4 = torch.nn.LeakyReLU()

        self.fcon_5 = torch.nn.Linear(
            in_features  = n_fcon_4,
            out_features = w_out
        )
        
        self.resetParams()


    def resetParams(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                    
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):

        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.nonl_1(x)
        
        x = self.pool_1(x)
        
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.nonl_2(x)
        
        x = self.pool_2(x)
        
        x = self.conv_3(x)
        x = self.norm_3(x)
        x = self.nonl_3(x)
        
        x = self.pool_3(x)

        x = x.contiguous().view(-1, n_conv_3 * (seq_len // 64))

        y_hat = self.fcon_4(x)
        y_hat = self.norm_4(y_hat)
        y_hat = self.nonl_4(y_hat)

        y_hat = self.fcon_5(y_hat)

        y_hat = y_hat.view(-1, self.w_out)

        return(y_hat)
        
