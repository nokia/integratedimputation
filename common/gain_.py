# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

# http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf

from defaults    import *

import utils

import numpy as np

import torch


# ==============================================================================
# Defs =========================================================================
def evaluate(
    net_ae,
    net_disc,
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
            out      = net_ae(x_mis_in)
            out_imp  = x_mis + ((1 - mask) * out)
            out_disc = net_disc(out_imp)
            
            acc_disc = torch.mean((torch.round(out_disc) == mask).double())

            logger.accumulate([acc_disc.item()], [4])
            
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
    alpha,
    iter_disc,
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
    
    net_disc = net_dict["net_disc"]
    opt_disc = opt_dict["opt_disc"]
    
    logger = utils.Logger(["l_ae_mis", "l_ae_rec", "l_disc", "acc_disc", "acc_eval_disc"], epoch_digits, out_folder, epochs_start != 0)

    for epoch in range(epochs_start, epochs_end):
        net_ae.train()
        net_disc.train()
        for i, data in enumerate(dataloader_train, 0):
            x, _ = data
            x    = x.to(device)
            
            # Introduce missingness, generate noisy input
            x_mis, mask = introduceMissingTrain(x)

            # Attach mask to input
            mask_in  = (mask * 2) - 1 # Centering the mask
            x_mis_in = torch.cat([x_mis, mask_in], dim = 1)
            
            # Discriminator ----------------------------------------------------
            if introduceMissingTrain.missing_rate is not None and introduceMissingTrain.missing_rate != 0:
                with torch.no_grad():
                    out      = net_ae(x_mis_in)
                    out_imp  = x_mis + ((1 - mask) * out)

                # Random sample/shuffle inputs
                perm_both = torch.cat([x.unsqueeze(3), out_imp.unsqueeze(3)], dim = 3)
                perm_mask = (torch.rand_like(x) > 0.5).to(torch.long).unsqueeze(3)

                perm_real = torch.gather(perm_both, 3, perm_mask).squeeze(3)
                perm_fake = torch.gather(perm_both, 3, (1 - perm_mask)).squeeze(3)
                
                out_disc_real = net_disc(perm_real)
                out_disc_fake = net_disc(perm_fake)

                out_disc_both = torch.cat([out_disc_real.unsqueeze(3), out_disc_fake.unsqueeze(3)], dim = 3)
                out_disc_real = torch.gather(out_disc_both, 3, perm_mask).squeeze(3)
                out_disc_fake = torch.gather(out_disc_both, 3, (1 - perm_mask)).squeeze(3)
                
                # Losses
                l_disc_real = (1 - mask) * torch.log(out_disc_real + eps)
                l_disc_fake = (1 - mask) * torch.log(1 - out_disc_fake + eps)
                
                l_disc   = -torch.mean(l_disc_real + l_disc_fake)

                acc_disc = (
                    torch.sum((1 - mask) * (1 - torch.round(out_disc_fake))) +
                    torch.sum((1 - mask) * torch.round(out_disc_real))
                ) / (2 * torch.sum(1 - mask))
                
                opt_disc.zero_grad()
                l_disc.backward()
                opt_disc.step()
                
                logger.accumulate([l_disc.item(), acc_disc.item()], [2, 3])
            
            # AE ---------------------------------------------------------------
            if not (i % iter_disc):
                out      = net_ae(x_mis_in)
                out_imp  = x_mis + ((1 - mask) * out)
                
                out_disc_fake = net_disc(out_imp)
                
                l_ae_mis_fake = (1 - mask) * torch.log(out_disc_fake + eps)
                l_ae_mis      = -torch.mean(l_ae_mis_fake)
                
                l_ae_rec = torch.nn.MSELoss()(out, x)
                l_ae     = (alpha * l_ae_rec) + l_ae_mis
    
                opt_ae.zero_grad()
                l_ae.backward()
                opt_ae.step()

                logger.accumulate([l_ae_mis.item(), l_ae_rec.item()])

        # Eval
        if not ((epoch + 1) % eval_every):
            evaluate(net_ae, net_disc, introduceMissingTrain, dataloader_test, device, epoch, eval_acc, logger)
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
