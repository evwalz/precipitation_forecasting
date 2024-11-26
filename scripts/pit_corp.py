
import numpy as np
import pandas as pd

from corp_functions import reliabilitydiag
from helpers import *


def upit(obs, ens):
    pit = np.zeros(len(obs))
    m = ens.shape[1]
    for k in range(len(obs)):
        rank_min = np.sum(ens[k,] < obs[k])+1
        rank_max = np.sum(ens[k,] <= obs[k])+1
        if rank_min == rank_max:
            rank_i = rank_min
        else:
            a = np.arange(rank_min, rank_max + 1, 1)
            rank_i = np.random.choice(a, 1)
        
        V = np.random.uniform(0, 1, 1)
        pit[k] = (rank_i-1)/ (m+1) + V / (m+1)
    return pit


def compute_rel(model, season, data_dir):
    # logit, logit_base, hres, ensemble, emos, cnn, hybrid
    data_save = data_dir + '/results/'
    folds = np.arange(9)
    lsm = np.loadtxt(data_dir  + "/lsm.txt")

    pop_list = list()
    obs_list = list()
    if model == 'mpc':
        for fold in folds:
            ytrain, time_train = load_obs_time(data_dir, fold, mode = "train")
            if fold == 8:
                yval, time_test = load_obs_time(data_dir, fold, mode = "test")
            else:
                yval, time_test = load_obs_time(data_dir, fold, mode = "val")

            # array with time x lat x lon
            year_dim = yval.shape[0]
            ix0, ix1 = sel_season_indx(season, year_dim)
            time_test_season = time_test[ix0:ix1]
            unique_tts = np.unique(time_test_season)
                    
            i = 13
            j = 27
            ytrain_grid = ytrain[:, i, j] 
            yval_grid = yval[ix0:ix1, i, j]
            crps_ens = list()
            bs_ens = list()
            for val in unique_tts:
                ix_month = np.where(val == time_train)[0]
                month_ens = ytrain_grid[ix_month]
                month_prob = np.mean(month_ens > 0.2)
                yval_grid_month = yval_grid[val == time_test_season]
                ens_dim = len(yval_grid_month)
                #crps_ens.append(ps.crps_ensemble(yval_grid_month, ens_mat))
                yval_grid_month_bin = yval_grid_month > 0.2
                ens_prob = np.repeat(month_prob, ens_dim)
                pop_list.append(ens_prob)
                obs_list.append(yval_grid_month_bin)
                
        rel_object = reliabilitydiag(np.concatenate(pop_list), np.concatenate(obs_list))
    else:
        rel_object = None
        print('not yet implemented')
    return rel_object




def compute_pit(model, season, data_dir):
    # dim, dim_base, hres, ensemble, emos, cnn, hybrid
    data_save = data_dir + '/results/'
    folds = np.arange(9)
    lsm = np.loadtxt(data_dir  + "/lsm.txt")

    pit_list = list()
    if model == 'mpc':
        for fold in folds:
            ytrain, time_train = load_obs_time(data_dir, fold, mode = "train")
            if fold == 8:
                yval, time_test = load_obs_time(data_dir, fold, mode = "test")
            else:
                yval, time_test = load_obs_time(data_dir, fold, mode = "val")

            # array with time x lat x lon
            year_dim = yval.shape[0]
            ix0, ix1 = sel_season_indx(season, year_dim)
            time_test_season = time_test[ix0:ix1]
            unique_tts = np.unique(time_test_season)
                    
            i = 13
            j = 27
            ytrain_grid = ytrain[:, i, j] 
            yval_grid = yval[ix0:ix1, i, j]
            crps_ens = list()
            bs_ens = list()
            for val in unique_tts:
                ix_month = np.where(val == time_train)[0]
                month_ens = ytrain_grid[ix_month]
                yval_grid_month = yval_grid[val == time_test_season]
                ens_dim = len(yval_grid_month)    
                ens_mat = np.tile(month_ens, (ens_dim, 1))
                pits = upit(yval_grid_month, ens_mat)
                
                pit_list.append(pits)
        pit_vals = np.concatenate(pit_list)
    else:
        print('not yet implemented')
        pit_vals = None
    return pit_vals