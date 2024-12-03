
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression

from corp_functions import reliabilitydiag
from helpers import *
from isodisreg import idr
import xarray as xr
from scipy.interpolate import interp1d
import random
import os


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
    # logit, logit_base, hres, ensemble, cnn, hybrid
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
    elif model == 'logit' or model == 'logit_base':
        if model == 'logit':
            feature_set = 'v2+time'
        else:
            feature_set = 'v1+time' 
        add_time = True
        model = LogisticRegression(max_iter = 500)
        for fold in folds:
            paths = PrecipitationDataPaths()
            feature_set_train, feature_set_test = select_data_subset(paths=paths, version=feature_set, fold=fold)
            Xtrain_prev, ytrain = load_and_concat(data_dir, fold, feature_set_train, add_time = add_time, mode = "train")
            if fold == 8:
                Xval_prev, yval = load_and_concat(data_dir, fold, feature_set_test, add_time = add_time, mode = "test")
            else:
                Xval_prev, yval = load_and_concat(data_dir, fold, feature_set_train, add_time = add_time, mode = "val")
    
            scaler_inputs = PerFeatureMeanStdScaler()

            Xtrain = scaler_inputs.fit_transform(Xtrain_prev)
            Xval = scaler_inputs.transform(Xval_prev)

            year_dim = yval.shape[0]
            ix0, ix1 = sel_season_indx(season, year_dim)
            i = 13
            j = 27
            obs_train_bin = ytrain[:, i, j] > 0.2
            obs_test_bin = yval[ix0:ix1, i, j] > 0.2
            if np.sum(obs_train_bin) == 0:
                probs_rain = np.zeros(len(obs_test_bin))
            else:
                clf = model.fit(Xtrain[:, :, i, j], obs_train_bin)
                probs_rain = clf.predict_proba(Xval[ix0:ix1 , :, i, j])[:, 1]
            pop_list.append(probs_rain)
            obs_list.append(obs_test_bin)   
        rel_object = reliabilitydiag(np.concatenate(pop_list), np.concatenate(obs_list))
    elif model == 'cnn':
        # check if forecast data are available:
        file_path = data_dir + '/forecasts/cnn_fct/subset_val_preds_v2+time_fold0.nc'
        if not os.path.isfile(file_path):
            raise ValueError('CNN forecast data does not exists')
        feature_set = 'v2+time'
        add_time = True
        for fold in folds:
            val_preds = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_val_preds_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            val_tar = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_val_target_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            train_prds = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_train_preds_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            train_tar = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_train_target_'+str(feature_set)+'_fold'+str(fold)+'.nc')

            year_dim = val_tar.dims['time']
            ix0, ix1 = sel_season_indx(season, year_dim)
            i = 13
            j = 27
            lat = i
            lon = j-25
            train_fct = train_prds.sel(lat = lat, lon = lon).train_preds.values
            train_obs = train_tar.sel(lat = lat, lon = lon).train_tar.values
            val_fct = val_preds.sel(lat = lat, lon = lon).val_preds.values[ix0:ix1]
            val_obs = val_tar.sel(lat = lat, lon = lon).val_tar.values[ix0:ix1]
            idr_output = idr(train_obs, pd.DataFrame({'fore': train_fct}, columns = ['fore']))
            pred_idr = idr_output.predict(pd.DataFrame({'fore': val_fct}, columns = ['fore']))
            probs_rain = 1- pred_idr.cdf(0.2)
            pop_list.append(probs_rain)
            obs_list.append(val_obs > 0.2)   
        rel_object = reliabilitydiag(np.concatenate(pop_list), np.concatenate(obs_list))
    elif model == 'hybrid':
        # check if forecast data are available:
        file_path = data_dir + '/forecasts/hres_fct/HRES24_multi1000_precip_init00_66_reconcut_years_2001_2019.nc'
        if not os.path.isfile(file_path):
            raise ValueError('HRES forecast data does not exists')
        file_path2 = data_dir + '/forecasts/cnn_fct/subset_val_preds_v2+time_fold0.nc'
        if not os.path.isfile(file_path2):
            raise ValueError('CNN forecast data does not exists')
        hres_data =  xr.open_dataset(data_dir + '/forecasts/hres_fct/HRES24_multi1000_precip_init00_66_reconcut_years_2001_2019.nc')
        feature_set = 'v2+time'
        add_time = True
        for fold in folds:
            val_preds = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_val_preds_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            val_tar = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_val_target_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            train_prds = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_train_preds_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            train_tar = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_train_target_'+str(feature_set)+'_fold'+str(fold)+'.nc')

            
            date_train, date_test = hres_season_time(season, fold + 11)
            hres_data_train = hres_data.sel(time = date_train)
            hres_data_test = hres_data.sel(time = date_test)

            year_dim = val_tar.dims['time']
            ix0, ix1 = sel_season_indx(season, year_dim)
            i = 13
            j = 27
            lat = i
            lon = j-25
            train_fct = train_prds.sel(lat = lat, lon = lon).train_preds.values
            train_obs = train_tar.sel(lat = lat, lon = lon).train_tar.values
            val_fct = val_preds.sel(lat = lat, lon = lon).val_preds.values[ix0:ix1]
            val_obs = val_tar.sel(lat = lat, lon = lon).val_tar.values[ix0:ix1]
            train_hres = hres_data_train.sel(lat = lat, lon = lon).lsp.values
            val_hres = hres_data_test.sel(lat = lat, lon = lon).lsp.values
            train_ix0 = len(train_obs) - len(train_hres)
            
            idr_output = idr(train_obs, pd.DataFrame({'fore': train_fct}, columns = ['fore']))
            pred_idr = idr_output.predict(pd.DataFrame({'fore': val_fct}, columns = ['fore']))

            idr_output2 = idr(train_obs[train_ix0:], pd.DataFrame({'fore': train_hres}, columns = ['fore']))
            pred_idr2 = idr_output2.predict(pd.DataFrame({'fore': val_hres}, columns = ['fore']))

            probs_rain = 1- (0.5*(pred_idr.cdf(0.2) + pred_idr2.cdf(0.2)))
            pop_list.append(probs_rain)
            obs_list.append(val_obs > 0.2)   
        rel_object = reliabilitydiag(np.concatenate(pop_list), np.concatenate(obs_list))
    elif model == 'ensemble_pp':
        # check if forecast data are available:
        file_path = data_dir + '/forecasts/ensemble_fct/ens24_pop_reconcut_invertlat_mm_2006_2019.nc'
        if not os.path.isfile(file_path):
            raise ValueError('Ensemble forecast data does not exists')
        ecmwf_data_dir = data_dir + '/forecasts/ensemble_fct/'
        pop = xr.open_dataset(ecmwf_data_dir + 'ens24_pop_reconcut_invertlat_mm_2006_2019.nc')
        for fold in folds:
            ytrain = load_obs(data_dir, fold, mode = "train")
            dim_train = ytrain.shape[0]
            if fold == 8:
                yval = load_obs(data_dir, fold, mode = "test")
            else:
                yval = load_obs(data_dir, fold, mode = "val")

            date_train, date_test = ecmwf_season_time(season, fold + 11)
            pop_train = pop.sel(time = date_train)
            pop_test = pop.sel(time = date_test)

            dim_pop = pop_train.dims['time']
            new_dim = dim_train - dim_pop

            year_dim = yval.shape[0]
            ix0, ix1 = sel_season_indx(season, year_dim)

            i = 13
            j = 27
            lat = i
            lon = j-25

            probs_train_grid = pop_train.sel(lat = lat, lon = lon).tp.values
            probs_test_grid = pop_test.sel(lat = lat, lon = lon).tp.values
                
            obs_bin_train = ytrain[new_dim:, i, j] > 0.2
            obs_bin_test = yval[ix0:ix1, i, j] > 0.2

            fit = idr(y = 1*obs_bin_train, X = pd.DataFrame(probs_train_grid), progress = False)
            preds = fit.predict(pd.DataFrame(probs_test_grid))
            pop_test = 1-preds.cdf(0.2)
            
            obs_list.append(obs_bin_test)
            pop_list.append(pop_test)  
        rel_object = reliabilitydiag(np.concatenate(pop_list), np.concatenate(obs_list)) 
    elif model == 'hres':
        # check if forecast data are available:
        file_path = data_dir + '/forecasts/hres_fct/HRES24_multi1000_precip_init00_66_reconcut_years_2001_2019.nc'
        if not os.path.isfile(file_path):
            raise ValueError('HRES forecast data does not exists')
        hres_data =  xr.open_dataset(data_dir + '/forecasts/hres_fct/HRES24_multi1000_precip_init00_66_reconcut_years_2001_2019.nc')
        for fold in folds:
            ytrain = load_obs(data_dir, fold, mode = "train")
            dim_train = ytrain.shape[0]
    
            if fold == 8:
                yval = load_obs(data_dir, fold, mode = "test")
            else:
                yval = load_obs(data_dir, fold, mode = "val")

            date_train, date_test = hres_season_time(season, fold + 11)
            hres_data_train = hres_data.sel(time = date_train)
            hres_data_test = hres_data.sel(time = date_test)

            year_dim = yval.shape[0]
            ix0, ix1 = sel_season_indx(season, year_dim)
            i = 13
            j = 27
            lat = i
            lon = j-25
            train_hres = hres_data_train.sel(lat = lat, lon = lon).lsp.values
            val_hres = hres_data_test.sel(lat = lat, lon = lon).lsp.values
            train_ix0 = dim_train - len(train_hres)
                
            idr_output = idr(ytrain[train_ix0:, i, j], pd.DataFrame({'fore': train_hres}, columns = ['fore']))
            pred_idr = idr_output.predict(pd.DataFrame({'fore': val_hres}, columns = ['fore']))

            probs_rain = 1- pred_idr.cdf(0.2)
            pop_list.append(probs_rain)
            obs_list.append(yval[ix0:ix1, i, j] > 0.2) 
        rel_object = reliabilitydiag(np.concatenate(pop_list), np.concatenate(obs_list))
    elif model == 'ensemble':
        file_path = data_dir + '/forecasts/ensemble_fct/ens_0.nc'
        if not os.path.isfile(file_path):
            raise ValueError('Ensemble forecast data does not exists')
        # check if forecast data are available:
        ecmwf_data_dir = data_dir + '/forecasts/ensemble_fct/'
        for fold in folds:
            if fold == 8:
                yval = load_obs(data_dir, fold, mode = "test")
            else:
                yval = load_obs(data_dir, fold, mode = "val")
            year_dim = yval.shape[0]
            ix0, ix1 = sel_season_indx(season, year_dim)
            ens = xr.open_dataset(ecmwf_data_dir + 'ens_'+str(fold)+'.nc')
            i = 13
            j = 27
            lat = i
            lon = j-25
            ens_test = ens.sel(lat = lat, lon = lon)['var'].values[ix0:ix1, :]
            pop_test = np.mean(ens_test>0.2, axis = 1)
            ygrid = yval[ix0:ix1, i, j]
            obs_list.append(ygrid > 0.2)
            pop_list.append(pop_test)  
        rel_object = reliabilitydiag(np.concatenate(pop_list), np.concatenate(obs_list))               
    else:
        rel_object = None
        print('not yet implemented')
    return rel_object

def pit_av_idr(y, points, ecdf_list, randomize = True, seed = None):
    ly = y.size
    def pit0 (ecdf, y):
        return(interp1d(x = np.hstack([np.min(points), points]), y = np.hstack([0,ecdf]), kind='previous', fill_value="extrapolate")(y))
    
    pitVals = np.array(list(map(pit0, ecdf_list, list(y))))
    if randomize:
        sel = [len(x) for x in ecdf_list]
        sel = np.where(np.array(sel) > 1)[0]
        if not any(sel):
            eps = 1
        else :
            preds_sel = [ecdf_list[i] for i in sel]
            eps = np.min([np.min(np.diff(points)) for x in preds_sel])
        lowerPitVals = np.array(list(map(pit0, ecdf_list, y-eps*0.5)))
        if seed is not None:
            random.seed(seed)
        sel = lowerPitVals < pitVals
        if any(sel):
            pitVals[sel] = np.random.uniform(low = lowerPitVals[sel], high = pitVals[sel], size = np.sum(sel)) 
    return(pitVals)

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
    elif model == 'dim' or model == 'dim_base':
        if model == 'dim':
            feature_set = 'v2+time'
        else:
            feature_set = 'v1+time' 
        add_time = True 
        model = LinearRegression()     
        for fold in folds:
            paths = PrecipitationDataPaths()
            feature_set_train, feature_set_test = select_data_subset(paths=paths, version=feature_set, fold=fold)
            Xtrain_prev, ytrain = load_and_concat(data_dir, fold, feature_set_train, add_time = add_time, mode = "train")
            if fold == 8:
                Xval_prev, yval = load_and_concat(data_dir, fold, feature_set_test, add_time = add_time, mode = "test")
            else:
                Xval_prev, yval = load_and_concat(data_dir, fold, feature_set_train, add_time = add_time, mode = "val")
    
            scaler_inputs = PerFeatureMeanStdScaler()

            Xtrain = scaler_inputs.fit_transform(Xtrain_prev)
            Xval = scaler_inputs.transform(Xval_prev)

            year_dim = yval.shape[0]
            ix0, ix1 = sel_season_indx(season, year_dim)
            i = 13
            j = 27   
            ytrain_log = np.log(ytrain[:, i, j] + 0.001)
            reg = model.fit(Xtrain[:, :, i, j], ytrain_log)
            reg_train =  np.exp(reg.predict(Xtrain[:, :, i, j])) - 0.001
            reg_valid = np.exp(reg.predict(Xval[ix0:ix1, :, i, j])) - 0.001
 
            idr_output = idr(ytrain[:, i, j], pd.DataFrame({'fore': reg_train}, columns = ['fore']))
            pred_idr = idr_output.predict(pd.DataFrame({'fore': reg_valid}, columns = ['fore']))
            pit_list.append(pred_idr.pit(yval[ix0:ix1, i, j]))

        pit_vals = np.concatenate(pit_list)
    elif model == 'hres':
        file_path = data_dir + '/forecasts/hres_fct/HRES24_multi1000_precip_init00_66_reconcut_years_2001_2019.nc'
        if not os.path.isfile(file_path):
            raise ValueError('HRES forecast data does not exists')
        hres_data =  xr.open_dataset(data_dir + '/forecasts/hres_fct/HRES24_multi1000_precip_init00_66_reconcut_years_2001_2019.nc')
        for fold in folds:
            ytrain = load_obs(data_dir, fold, mode = "train")
            dim_train = ytrain.shape[0]
    
            if fold == 8:
                yval = load_obs(data_dir, fold, mode = "test")
            else:
                yval = load_obs(data_dir, fold, mode = "val")

            date_train, date_test = hres_season_time(season, fold + 11)
            hres_data_train = hres_data.sel(time = date_train)
            hres_data_test = hres_data.sel(time = date_test)

            year_dim = yval.shape[0]
            ix0, ix1 = sel_season_indx(season, year_dim)
            i = 13
            j = 27
            lat = i
            lon = j-25
            train_hres = hres_data_train.sel(lat = lat, lon = lon).lsp.values
            val_hres = hres_data_test.sel(lat = lat, lon = lon).lsp.values
            train_ix0 = dim_train - len(train_hres)
                
            idr_output = idr(ytrain[train_ix0:, i, j], pd.DataFrame({'fore': train_hres}, columns = ['fore']))
            pred_idr = idr_output.predict(pd.DataFrame({'fore': val_hres}, columns = ['fore']))
            pit_list.append(pred_idr.pit(yval[ix0:ix1, i, j]))
        pit_vals = np.concatenate(pit_list)
    elif model == 'cnn':
        file_path = data_dir + '/forecasts/cnn_fct/subset_val_preds_v2+time_fold0.nc'
        if not os.path.isfile(file_path):
            raise ValueError('CNN forecast data does not exists')
        feature_set = 'v2+time'
        add_time = True
        for fold in folds:
            val_preds = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_val_preds_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            val_tar = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_val_target_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            train_prds = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_train_preds_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            train_tar = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_train_target_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            year_dim = val_tar.dims['time']
            ix0, ix1 = sel_season_indx(season, year_dim)
            i = 13
            j = 27
            lat = i
            lon = j-25
            train_fct = train_prds.sel(lat = lat, lon = lon).train_preds.values
            train_obs = train_tar.sel(lat = lat, lon = lon).train_tar.values
            val_fct = val_preds.sel(lat = lat, lon = lon).val_preds.values[ix0:ix1]
            val_obs = val_tar.sel(lat = lat, lon = lon).val_tar.values[ix0:ix1]
            idr_output = idr(train_obs, pd.DataFrame({'fore': train_fct}, columns = ['fore']))
            pred_idr = idr_output.predict(pd.DataFrame({'fore': val_fct}, columns = ['fore']))
            pit_list.append(pred_idr.pit(val_obs))  
        pit_vals = np.concatenate(pit_list) 
    elif model == 'hybrid':
        file_path = data_dir + '/forecasts/hres_fct/HRES24_multi1000_precip_init00_66_reconcut_years_2001_2019.nc'
        if not os.path.isfile(file_path):
            raise ValueError('HRES forecast data does not exists')
        file_path2 = data_dir + '/forecasts/cnn_fct/subset_val_preds_v2+time_fold0.nc'
        if not os.path.isfile(file_path2):
            raise ValueError('CNN forecast data does not exists')      
        hres_data =  xr.open_dataset(data_dir + '/forecasts/hres_fct/HRES24_multi1000_precip_init00_66_reconcut_years_2001_2019.nc')
        feature_set = 'v2+time'
        add_time = True
        for fold in folds:
            val_preds = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_val_preds_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            val_tar = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_val_target_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            train_prds = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_train_preds_'+str(feature_set)+'_fold'+str(fold)+'.nc')
            train_tar = xr.open_dataset(data_dir + '/forecasts/cnn_fct/subset_train_target_'+str(feature_set)+'_fold'+str(fold)+'.nc')

            
            date_train, date_test = hres_season_time(season, fold + 11)
            hres_data_train = hres_data.sel(time = date_train)
            hres_data_test = hres_data.sel(time = date_test)

            year_dim = val_tar.dims['time']
            ix0, ix1 = sel_season_indx(season, year_dim)
            i = 13
            j = 27
            lat = i
            lon = j-25
            train_fct = train_prds.sel(lat = lat, lon = lon).train_preds.values
            train_obs = train_tar.sel(lat = lat, lon = lon).train_tar.values
            val_fct = val_preds.sel(lat = lat, lon = lon).val_preds.values[ix0:ix1]
            val_obs = val_tar.sel(lat = lat, lon = lon).val_tar.values[ix0:ix1]
            train_hres = hres_data_train.sel(lat = lat, lon = lon).lsp.values
            val_hres = hres_data_test.sel(lat = lat, lon = lon).lsp.values
            train_ix0 = len(train_obs) - len(train_hres)
            
            idr_output = idr(train_obs, pd.DataFrame({'fore': train_fct}, columns = ['fore']))
            pred_idr = idr_output.predict(pd.DataFrame({'fore': val_fct}, columns = ['fore']))

            idr_output2 = idr(train_obs[train_ix0:], pd.DataFrame({'fore': train_hres}, columns = ['fore']))
            pred_idr2 = idr_output2.predict(pd.DataFrame({'fore': val_hres}, columns = ['fore']))
            
            grid_idr = np.sort(np.unique(train_obs))
            preds_average =  0.5*(pred_idr.cdf(grid_idr) + pred_idr2.cdf(grid_idr))
            preds_average_list = preds_average.tolist()
            pit_vals = pit_av_idr(val_obs, grid_idr, preds_average_list)

            pit_list.append(pit_vals)  
            #probs_rain = 1- (0.5*(pred_idr.cdf(0.2) + pred_idr2.cdf(0.2)))
        pit_vals = np.concatenate(pit_list)
    elif model == 'ensemble':
        file_path = data_dir + '/forecasts/ensemble_fct/ens_0.nc'
        if not os.path.isfile(file_path):
            raise ValueError('Ensemble forecast data does not exists')
        ecmwf_data_dir = data_dir + '/forecasts/ensemble_fct/'
        for fold in folds:
            if fold == 8:
                yval = load_obs(data_dir, fold, mode = "test")
            else:
                yval = load_obs(data_dir, fold, mode = "val")
            year_dim = yval.shape[0]
            ix0, ix1 = sel_season_indx(season, year_dim)
            ens = xr.open_dataset(ecmwf_data_dir + 'ens_'+str(fold)+'.nc')
            i = 13
            j = 27
            lat = i
            lon = j-25
            ens_test = ens.sel(lat = lat, lon = lon)['var'].values[ix0:ix1, :]
            ygrid = yval[ix0:ix1, i, j]
            pits = upit(ygrid, ens_test)
            pit_list.append(pits)
        pit_vals = np.concatenate(pit_list)
    elif model == 'emos':
        pit_vals = np.loadtxt(data_dir + '/results/prev_results_emos/pit_emos_niamey.txt')
    else:
        raise ValueError('model not specified')
    return pit_vals