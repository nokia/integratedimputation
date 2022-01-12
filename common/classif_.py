# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from defaults    import *

import utils 

import numpy as np
import math
import itertools

import torch

# ==============================================================================
# Defs =========================================================================
def evaluate(
    net_class,
    introduceMissingTrain,
    dataloader_test,
    device,
    logger
):
    with torch.no_grad():
        net_class.eval()
        for i, data in enumerate(dataloader_test, 0):
            x, labs = data
            x       = x.to(device)
            labs    = labs.to(device)

            # Introduce missingness
            x_mis, mask = introduceMissingTrain(x)

            # Attach mask to input
            mask_in  = (mask * 2) - 1 # Centering the mask
            x_mis_in = torch.cat([x_mis, mask_in], dim = 1)

            # Main forward -----------------------------------------------------
            out = net_class(x_mis_in)
            l_class = torch.nn.CrossEntropyLoss()(out, labs)

            logger.accumulate([l_class], [1])
            
    for i_mr, missing_rate in enumerate(missing_rates_eval):
        for j_mt, missing_type in enumerate(missing_types_eval):
                
            if missing_type == 'seq':
                introduceMissingEval = utils.IntroduceMissingSeq(missing_rate)
            else:
                introduceMissingEval = utils.IntroduceMissing(missing_rate)

            with torch.no_grad():
                net_class.eval()
                for i, data in enumerate(dataloader_test, 0):
                    x, labs = data
                    x       = x.to(device)
                    labs    = labs.to(device)
    
                    # Introduce missingness
                    x_mis, mask = introduceMissingEval(x)
    
                    # Attach mask to input
                    mask_in  = (mask * 2) - 1 # Centering the mask
                    x_mis_in = torch.cat([x_mis, mask_in], dim = 1)
    
                    # Main forward
                    ass = torch.argmax(net_class(x_mis_in), 1, keepdim = False)
                    acc = (labs == ass).sum().item() / ass.shape[0]
    
                    logger.accumulate([acc], [3 + (2*i_mr + j_mt)])

def train(
    introduceMissingTrain,
    net_dict,
    opt_dict,
    dataloader_train,
    dataloader_test,
    device,
    eval_every,
    out_folder,
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
        
    net_class = net_dict["net_class"]
    opt_class = opt_dict["opt_class"]

    logger = utils.Logger(["l_train", "l_eval", "acc_train"] + [f"acc_{m_type}_" + str(m_rate).replace(".", "") for m_rate, m_type in list(itertools.product(missing_rates_eval, missing_types_eval))], epoch_digits, out_folder, epochs_start != 0)

    for epoch in range(epochs_start, epochs_end):
        net_class.train()
        for i, data in enumerate(dataloader_train, 0):
            x, labs = data
            x       = x.to(device)
            labs    = labs.to(device)
            
            # Introduce missingness
            x_mis, mask = introduceMissingTrain(x)
            
            # Attach mask to input
            mask_in  = (mask * 2) - 1 # Centering the mask
            x_mis_in = torch.cat([x_mis, mask_in], dim = 1)

            # Main forward -----------------------------------------------------
            out = net_class(x_mis_in)

            # Backprop ---------------------------------------------------------
            l_class = torch.nn.CrossEntropyLoss()(out, labs)
            
            opt_class.zero_grad()
            l_class.backward()
            opt_class.step()
            
            # Logging ----------------------------------------------------------
            ass = torch.argmax(out, 1, keepdim = False)
            acc = (labs == ass).sum().item() / ass.shape[0]

            logger.accumulate([l_class.item(), acc], [0, 2])

        # Eval
        if not ((epoch + 1) % eval_every):
            evaluate(net_class, introduceMissingTrain, dataloader_test, device, logger)
            logger.log(epoch + 1)

    # Save
    utils.saveAll(
        net_dict,
        opt_dict,
        epoch + 1,
        epoch_digits,
        out_folder
    )

    print('Finished Training')
