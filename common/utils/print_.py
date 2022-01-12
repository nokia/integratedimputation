# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

def padLen(num, digits = 3):
    num_str = str(num)
    pad_len = (digits - len(num_str))
    
    return(num_str, pad_len)

def padSpaces(num, digits = 3):
    num_str, pad_len = padLen(num, digits)

    return('[' + num_str + ']' + (pad_len * ' '))
    
def padZeros(num, digits = 3):
    num_str, pad_len = padLen(num, digits)
    
    return((pad_len * '0') + num_str)
