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
n_conv_1 = 512
n_conv_2 = 256
n_conv_3 = 128


# ==============================================================================
# Module definitions ===========================================================
class NetAEConvTrans(utils.DmModule):
    def __init__(self, w_in, w_out):
        super().__init__()
        
        self.w_out = w_out

        # Encoder
        self.enc_conv_1 = torch.nn.ConvTranspose1d(w_in, n_conv_1, 3, stride = 1, padding = 1)
        self.enc_nonl_1 = torch.nn.LeakyReLU()
        
        self.enc_samp_12 = torch.nn.Upsample(scale_factor = 2)

        self.enc_conv_2 = torch.nn.ConvTranspose1d(n_conv_1, n_conv_2, 3, stride = 1, padding = 1)
        self.enc_norm_2 = torch.nn.BatchNorm1d(n_conv_2)
        self.enc_nonl_2 = torch.nn.LeakyReLU()
        
        self.enc_samp_23 = torch.nn.Upsample(scale_factor = 2)
        
        self.enc_conv_3 = torch.nn.ConvTranspose1d(n_conv_2, n_conv_3, 3, stride = 1, padding = 1)
        self.enc_nonl_3 = torch.nn.LeakyReLU()

        # Decoder
        self.dec_conv_3 = torch.nn.Conv1d(n_conv_3, n_conv_2, 3, stride = 1, padding = 1)
        self.dec_nonl_3 = torch.nn.LeakyReLU()
        
        self.dec_pool_32 = torch.nn.MaxPool1d(2, stride = 2)
        
        self.dec_conv_2 = torch.nn.Conv1d(n_conv_2, n_conv_1, 3, stride = 1, padding = 1)
        self.dec_norm_2 = torch.nn.BatchNorm1d(n_conv_1)
        self.dec_nonl_2 = torch.nn.LeakyReLU()
        
        self.dec_pool_21 = torch.nn.MaxPool1d(2, stride = 2)
        
        self.dec_conv_1 = torch.nn.Conv1d(n_conv_1, self.w_out, 3, stride = 1, padding = 1)
        
        self.resetParams()

        
    def resetParams(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                    
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
                

    def encode(self, x):
        x = self.enc_conv_1(x)
        x = self.enc_nonl_1(x)
        
        x = self.enc_samp_12(x)
        
        x = self.enc_conv_2(x)
        x = self.enc_norm_2(x)
        x = self.enc_nonl_2(x)
        
        x = self.enc_samp_23(x)
        
        x = self.enc_conv_3(x)
        x = self.enc_nonl_3(x)

        return(x)
        
        
    def decode(self, x):
        
        x = self.dec_conv_3(x)
        x = self.dec_nonl_3(x)
        
        x = self.dec_pool_32(x)
        
        x = self.dec_conv_2(x)
        x = self.dec_norm_2(x)
        x = self.dec_nonl_2(x)
        
        x = self.dec_pool_21(x)
        
        x = self.dec_conv_1(x)

        return(x)
        
        
    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        
        return(dec)
