# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.nn import Module

from collections import OrderedDict
import functools
import itertools

from torch.nn.parameter import Parameter
import torch.utils.hooks as hooks

# Device-managed or deepMeans module, that holds the currently associated device
# in a field in the class. Every module inherits from this class.
class DmModule(Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu:0")
        
    def to(self, *args, **kwargs):
        device, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self._to_device(device)
        
        return(super().to(*args, **kwargs))
        
    def _to_device(self, device):
        self.device = device
        
        for module in self.children():
            if isinstance(module, DmModule):
                module._to_device(device)
