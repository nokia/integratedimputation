# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

# https://arxiv.org/pdf/1705.02737.pdf

from defaults    import *

import utils

import numpy as np

import torch


# ==============================================================================
# Defs =========================================================================
def evaluate(
    net_ae,
    introduceMissingTrain,
    dataloader_test,
    device,
    epoch,
    eval_acc,
    logger
):
    with torch.no_grad():
        net_ae.eval()
        for i, data in enumerate(dataloader_test, 0):
            x, _ = data
            x    = x.to(device)

            # Introduce missingness
            x_mis, mask = introduceMissingTrain(x)

            # Attach mask to input
            mask_in  = (mask * 2) - 1 # Centering the mask
            x_mis_in = torch.cat([x_mis, mask_in], dim = 1)

            # Main forward
            out = net_ae(x_mis_in)
            
            l_mse = torch.nn.MSELoss()(out, x)

            logger.accumulate([l_mse.item()], [1])
            
    for i_mr, missing_rate in enumerate(missing_rates_eval):
        for j_mt, missing_type in enumerate(missing_types_eval):
                
            if missing_type == 'seq':
                introduceMissingEval = utils.IntroduceMissingSeq(missing_rate)
            else:
                introduceMissingEval = utils.IntroduceMissing(missing_rate)
            
            with torch.no_grad():
                net_ae.eval()
                for i, data in enumerate(dataloader_test, 0):
                    x, l = data
                    x    = x.to(device)
    
                    # Introduce missingness
                    x_mis, mask = introduceMissingEval(x)
    
                    # Attach mask to input
                    mask_in  = (mask * 2) - 1 # Centering the mask
                    x_mis_in = torch.cat([x_mis, mask_in], dim = 1)
        
                    # Main forward
                    out = net_ae(x_mis_in)
                    
                    eval_acc.accumulate(x, x_mis, mask, out, l, epoch+1, 2*i_mr + j_mt)
                
def train(
    introduceMissingTrain,
    net_dict,
    opt_dict,
    dataloader_train,
    dataloader_test,
    device,
    eval_every,
    out_folder,
    eval_acc,
    epochs_end   = 10, 
    epochs_start = 0
):
    
    if epochs_start != 0:
        utils.loadAll(
            net_dict,
            opt_dict,
            epochs_start,
            epoch_digits,
            out_folder
        )
        
    net_ae = net_dict["net_ae"]
    opt_ae = opt_dict["opt_ae"]
    
    logger = utils.Logger(["l_train", "l_eval"], epoch_digits, out_folder, epochs_start != 0)

    for epoch in range(epochs_start, epochs_end):
        net_ae.train()
        for i, data in enumerate(dataloader_train, 0):
            x, _ = data
            x    = x.to(device)
            
            # Introduce missingness
            x_mis, mask = introduceMissingTrain(x)

            # Attach mask to input
            mask_in  = (mask * 2) - 1 # Centering the mask
            x_mis_in = torch.cat([x_mis, mask_in], dim = 1)

            # Main forward -----------------------------------------------------
            out = net_ae(x_mis_in)

            # Backprop ---------------------------------------------------------
            l_ae = torch.nn.MSELoss()(out, x)
            
            opt_ae.zero_grad()
            l_ae.backward()
            opt_ae.step()

            logger.accumulate([l_ae.item()])

        # Eval
        if not ((epoch + 1) % eval_every):
            evaluate(net_ae, introduceMissingTrain, dataloader_test, device, epoch, eval_acc, logger)
            logger.log(epoch + 1)
            eval_acc.log(epoch + 1)

    # Save
    utils.saveAll(
        net_dict,
        opt_dict,
        epoch + 1,
        epoch_digits,
        out_folder
    )

    print('Finished Training')
