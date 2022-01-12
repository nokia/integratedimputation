#!/usr/bin/env python3

# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import sys

sys.path.append('./common/')
from defaults    import *

import utils

import subprocess

import time
from datetime import datetime

import argparse

# ==============================================================================
# Settings =====================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--print_only',   default = False)
parser.add_argument('--missing_type', default = "ran") # "ran" or "seq"
args = parser.parse_args()

test         = args.test
missing_type = args.missing_type

# Add to this list if you have more than 1 GPUs in your system, like:
# gpu_list  = [0, 1]
gpu_list = [0]

algorithms_list = []
algorithms_list.append(("classif",    ["fixm", "varm"]))

algorithms_list.append(("mean",       [None]))
algorithms_list.append(("mice",       [None]))
algorithms_list.append(("knn",        [None]))
algorithms_list.append(("missforest", [None]))

algorithms_list.append(("mida",       ["varm"]))
algorithms_list.append(("gain",       ["varm"]))
algorithms_list.append(("wgain",      ["varm"]))

# algorithms_list.append(("mida",       ["fixm", "varm"]))
# algorithms_list.append(("gain",       ["fixm", "varm"]))
# algorithms_list.append(("wgain",      ["fixm", "varm"]))

# ==============================================================================
# Definitions ==================================================================
def evaluate(folds_end = folds, folds_start = 0):
    task_desc_list = []

    for alg_name, variants in algorithms_list:
        for variant in variants:
            if variant == 'fixm':
                for missing_rate in missing_rates_train:
                    for fold in range(folds_start, folds_end):
                        task_desc_list.append((alg_name, variant, fold, missing_rate))
            else:
                for fold in range(folds_start, folds_end):
                    task_desc_list.append((alg_name, variant, fold))

    # --------------------------------------------------------------------------

    with open('./evals_log.txt', 'w') as log:
        current_time = datetime.now().strftime("%H:%M:%S")
        log_string   = f'[{current_time}] Started'
        log.write(log_string + '\n')
        print(log_string)

    # --------------------------------------------------------------------------

    workers = [None for i in range(len(gpu_list))]
    while True:
        finished = False

        for i, task in enumerate(workers):
            if task is None or task.poll() is not None:
                if len(task_desc_list) == 0:
                    finished = True

                else:
                    finished = False
                    task_desc = task_desc_list.pop(0)

                    alg_name, variant, fold, *add_args = task_desc

                    home_folder = f'./evals/{alg_name}'
                    add_args_dict = {}

                    if variant == None:
                        script_name = f'./{alg_name}.py'
                        out_folder  = f'./out_{utils.padZeros(fold, fold_digits)}'

                    elif variant == "fixm":
                        missing_rate = add_args[0]
                        missing_str = str(missing_rate).replace('.','')

                        script_name = f'./{alg_name}_{variant}.py'
                        out_folder  = f'./out_fixm_{missing_str}_{missing_type}_{utils.padZeros(fold, fold_digits)}'

                        add_args_dict["missing_rate_train"] = missing_rate
                        add_args_dict["missing_type"] = missing_type

                    elif variant == "varm":
                        script_name = f'./{alg_name}_{variant}.py'
                        out_folder  = f'./out_varm_{missing_type}_{utils.padZeros(fold, fold_digits)}'
                        
                        add_args_dict["missing_type"] = missing_type

                    # ----------------------------------------------------------

                    cmd_string = f'{script_name} --fold {fold} --out_folder {out_folder} --gpu_id {gpu_list[i]}'
                    
                    for arg_name, arg_val in add_args_dict.items():
                        cmd_string += f' --{arg_name} {arg_val}'

                    if not test:
                        with utils.cd(home_folder):
                            workers[i] = subprocess.Popen(cmd_string, shell = True)

                    with open('./evals_log.txt', 'a') as log:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        log_string   = f'[{current_time}] {home_folder} {script_name} {out_folder}'
                        log.write(log_string + '\n')
                        print(log_string)

        if finished:
            break
            
        if not test:
            time.sleep(10)

    # --------------------------------------------------------------------------

    with open('./evals_log.txt', 'a') as log:
        current_time = datetime.now().strftime("%H:%M:%S")
        log_string   = f'[{current_time}] Finished'
        log.write(log_string + '\n')
        print(log_string)


# ==============================================================================
# Calls ========================================================================
evaluate()
