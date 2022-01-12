#!/usr/bin/env python3

# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pandas as pd
import numpy as np

import sys
sys.path.append('./common/')
from defaults    import *

import glob
import itertools

from functools import partial

import argparse

# ==============================================================================
# Settings =====================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--missing_type',            default = "ran") # "ran" or "seq"
parser.add_argument('--match_eval_missing_type', default = True)
args = parser.parse_args()

missing_type  = args.missing_type

if args.match_eval_missing_type:
    missing_types_eval = [missing_type]

classif_variants = ["fixm", "varm"]

impute_list = []
impute_list.append(("mean",       [None]))
impute_list.append(("mice",       [None]))
impute_list.append(("knn",        [None]))
impute_list.append(("missforest", [None]))
impute_list.append(("mida",       ["varm"]))
impute_list.append(("gain",       ["varm"]))
impute_list.append(("wgain",      ["varm"]))

# ==============================================================================
# Definitions ==================================================================
def _accessAcc(folder, log_name = 'log.txt'):
    log_pd     = pd.read_csv(f'{folder}/{log_name}', delimiter = ',')
    metrics_np = log_pd[[f"acc_{m_type}_" + str(m_rate).replace(".", "") for m_rate, m_type in list(itertools.product(missing_rates_eval, missing_types_eval))]].values
    
    return(metrics_np[-1:, :])
    
    
def _accessMse(folder, log_name = 'log_mse.txt'):
    log_pd     = pd.read_csv(f'{folder}/{log_name}', delimiter = ',')
    metrics_np = log_pd[[f"mse_{m_type}_" + str(m_rate).replace(".", "") for m_rate, m_type in list(itertools.product(missing_rates_eval, missing_types_eval))]].values

    return(metrics_np[-1:, :])


def _accesFolds(folders_list, access_fun):
    metric_fold_avg_list = []
    for folder in folders_list:
        metric_fold_avg_list.append(access_fun(folder))

    metric_fold_avg_np = np.concatenate(metric_fold_avg_list, axis = 0)
    metric_fold_avg_np = np.mean(metric_fold_avg_np, axis = 0, keepdims = True)

    return(metric_fold_avg_np)

    
def _aggVariant(alg_name, variant, access_fun):
    rownames_list   = []
    metric_avg_list = []
    if variant == None:
        folders_list = glob.glob(f'./evals/{alg_name}/out_*')
        metric_avg_list.append(_accesFolds(folders_list, access_fun))
        rownames_list.append('n/a')
        
    elif variant == "fixm":
        for missing_rate in missing_rates_train:
            missing_str  = str(missing_rate).replace('.','')
            folders_list = glob.glob(f'./evals/{alg_name}/out_fixm_{missing_str}_{missing_type}*')
            metric_avg_list.append(_accesFolds(folders_list, access_fun))
            rownames_list.append(f'{missing_rate}')

    elif variant == "varm":
        folders_list = glob.glob(f'./evals/{alg_name}/out_varm_{missing_type}*')
        metric_avg_list.append(_accesFolds(folders_list, access_fun))
        rownames_list.append('var')
            
    metric_avg_np = np.concatenate(metric_avg_list, axis = 0)
    
    return(metric_avg_np, rownames_list)
    

def aggClassif():
        rownames_list_all   = []
        metric_avg_list_all = []

        for variant in classif_variants:
            metric_avg_np, rownames_list = _aggVariant('classif', variant, _accessAcc)
            metric_avg_list_all.append(metric_avg_np)
            rownames_list_all += rownames_list

        metric_avg_np = np.concatenate(metric_avg_list_all, axis = 0)

        colnames_list = [f"acc_{m_type}_" + str(m_rate).replace(".", "") for m_type, m_rate in list(itertools.product(missing_types_eval, missing_rates_eval))]
        metric_avg_pd = pd.DataFrame(
            metric_avg_np,
            index   = rownames_list_all,
            columns = colnames_list
        )
        
        print('classif')
        print(metric_avg_pd)
        print()
        print('----------------------------------------------------------')

def aggImpute():
    for alg_name, variants in impute_list:
        rownames_list_all = []
        acc_avg_list_all  = []
        mse_avg_list_all  = []

        # MSE
        for variant in variants:
            metric_avg_np, rownames_list = _aggVariant(alg_name, variant, _accessMse)
            mse_avg_list_all.append(metric_avg_np)
            rownames_list_all += rownames_list

        mse_avg_np = np.concatenate(mse_avg_list_all, axis = 0)
        
        colnames_list = [f"mse_{m_type}_" + str(m_rate).replace(".", "") for m_type, m_rate in list(itertools.product(missing_types_eval, missing_rates_eval))]
        mse_avg_pd = pd.DataFrame(
            mse_avg_np,
            index   = rownames_list_all,
            columns = colnames_list
        )
        
        print()
        print(alg_name, "mse")
        print(mse_avg_pd)

        # ACC
        for variant in variants:
            metric_avg_np, _ = _aggVariant(alg_name, variant, partial(_accessAcc, log_name = 'log_acc.txt'))
            acc_avg_list_all.append(metric_avg_np)

        acc_avg_np = np.concatenate(acc_avg_list_all, axis = 0)

        colnames_list = [f"acc_{m_type}_" + str(m_rate).replace(".", "") for m_type, m_rate in list(itertools.product(missing_types_eval, missing_rates_eval))]
        acc_avg_pd = pd.DataFrame(
            acc_avg_np,
            index   = rownames_list_all,
            columns = colnames_list
        )
        
        print()
        print(alg_name, "acc")
        print(acc_avg_pd)
        print()
        print('----------------------------------------------------------')
 
# ==============================================================================
# Calls ========================================================================
with pd.option_context('display.float_format', '{:0.6f}'.format):
    aggClassif()
    aggImpute()
