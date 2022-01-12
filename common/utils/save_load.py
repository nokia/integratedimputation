# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import torch

from .print_ import padZeros

def saveNet(net, model_name = "net_test", folder = "./out"):
    device = net.device
    net.to('cpu')
    
    file = folder + "/" + model_name + ".pth"
    torch.save(net.state_dict(), file)
    
    net.to(device)
    

def loadNet(net, model_name = "net_test", folder = "./out"):
    device = net.device
    net.to('cpu')
    
    file = folder + "/" + model_name + ".pth"
    net.load_state_dict(torch.load(file))
    
    net.to(device)
    
    
def saveOpt(opt, opt_name = "opt_test", folder = "./out"):
    file = folder + "/" + opt_name + ".pth"
    torch.save(opt.state_dict(), file)
    

def loadOpt(opt, opt_name = "opt_test", folder = "./out"):
    file = folder + "/" + opt_name + ".pth"
    state_dict = torch.load(file)
    
    device = opt.param_groups[0]['params'][0].device
    
    # Workaround for error which makes all tensor values 0 if tensor is moved between
    # different GPUs
    for k, v in state_dict['state'].items():
        for ki, vi in v.items():
            if torch.is_tensor(vi):
               state_dict['state'][k][ki] = state_dict['state'][k][ki].cpu().to(device)

    opt.load_state_dict(state_dict)
    

def saveAll(nets_dict, opts_dict, epoch, epoch_digits, folder):
    epoch_str = padZeros(epoch, epoch_digits)
    
    for name, net in nets_dict.items():
        saveNet(net, f"{name}_{epoch_str}", folder = folder)
        
    for name, opt in opts_dict.items():
        saveOpt(opt, f"{name}_{epoch_str}", folder = folder)
    

def loadAll(nets_dict, opts_dict, epoch, epoch_digits, folder):
    epoch_str = padZeros(epoch, epoch_digits)
    
    for name, net in nets_dict.items():
        loadNet(net, f"{name}_{epoch_str}", folder = folder)
        
    for name, opt in opts_dict.items():
        loadOpt(opt, f"{name}_{epoch_str}", folder = folder)
        
