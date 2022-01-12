# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import numpy as np

def gpuCount():
    query = [
        'nvidia-smi',
        '--list-gpus'
    ]
    res = subprocess.run(query, stdout=subprocess.PIPE).stdout.decode('ascii')[0:-1]
    
    # Count GPUs
    res = res.split("\n")
    
    return(len(res))

def gpuQuery(gpu_id = 0):
    query = [
        'nvidia-smi',
        '--query-gpu=gpu_bus_id,temperature.gpu,utilization.gpu,memory.used,clocks.current.sm',
        '--format=csv,noheader,nounits',
        f'--id={gpu_id}'
    ]
    res = subprocess.run(query, stdout=subprocess.PIPE).stdout.decode('ascii')[0:-1]
    
    # Create dict
    res = res.split(", ")
    return({
        "bus_id" : res[0],
        "temp"   : res[1],
        "util"   : res[2],
        "mem"    : res[3],
        "clk"    : res[4]
    })

def gpuAssign(overwrite = None, verbose = True):
    gpu_count = gpuCount()

    gpu_mems  = np.array([int(gpuQuery(gpu_id)['mem']) for gpu_id in range(gpu_count)])
    gpu_temps = np.array([int(gpuQuery(gpu_id)['temp']) for gpu_id in range(gpu_count)])

    if overwrite is not None:
        gpu_id = overwrite
    else:
        if(np.all(np.less(gpu_mems, 2000))):
            if(np.all(np.less(gpu_temps, 40))):
                gpu_id = np.random.randint(gpu_count)
            else:
                gpu_id = np.argmin(gpu_temps)
        else:
            gpu_id = np.argmin(gpu_mems)

    if verbose:
        if overwrite is not None:
            print(f'Assigned task to gpu_id: {gpu_id} (OVERWRITE!)')
        else:
            print(f'Assigned task to gpu_id: {gpu_id}')

    return(gpu_id)
