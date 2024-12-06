from dataclasses import dataclass
import pickle
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import genextreme
from scipy.special import gamma 
from scipy import stats  
from scipy.stats import spearmanr
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import transforms

@dataclass
class PrecipitationDataPaths:
    """Class to handle precipitation data."""

    path_pw_train = "predictors/train/tcwv_2000_2018.nc"
    path_cape_train = "predictors/train/cape_2000_2018.nc"
    path_cc_train = "predictors/train/cloudcover_2000_2018.nc"
    path_clwc_train = "predictors/train/cloudwater_2000_2018.nc"
    path_rh5_train = "predictors/train/rh500_2000_2018.nc"
    path_rh3_train = "predictors/train/rh300_2000_2018.nc"
    path_d2m_train = "predictors/train/d2m_2000_2018.nc"
    path_cin_train = "predictors/train/cin_2000_2018.nc"
    path_sh600_train = "predictors/train/spec_humid600_2000_2018.nc"
    path_sh925_train= "predictors/train/spec_humid925_2000_2018.nc"
    path_temp_train = "predictors/train/t2m_2000_2018.nc"
    path_kindx_train = "predictors/train/kindx_2000_2018.nc"
    path_sh7_train = "predictors/train/spec_humid700_2000_2018.nc"
    path_shear_train = "predictors/train/shear925_600_2000_2018.nc"
    path_stream_train = "predictors/train/stream_2000_2018.nc"
    path_vimd_train = "predictors/train/vimd_2000_2018.nc"
    path_pressure_tendency_train = "predictors/train/pressure_tendency_2000_2018.nc"
    path_precip_lag1_train = "predictors/train/precip_obs_lag_1_sel_2000_2018.nc"
    path_precip_lag2_train = "predictors/train/precip_obs_lag_2_sel_2000_2018.nc"
    path_precip_lag3_train = "predictors/train/precip_obs_lag_3_sel_2000_2018.nc"
    path_t850_train = "predictors/train/temp_850_2000_2018.nc"
    path_t500_train = "predictors/train/temp_500_2000_2018.nc"
    path_sh500_train = "predictors/train/spec_humid500_2000_2018.nc"

    path_pw_test = "predictors/test/tcwv_2019.nc"
    path_cape_test = "predictors/test/cape_2019.nc"
    path_cc_test = "predictors/test/cloudcover_2019.nc"
    path_clwc_test = "predictors/test/cloudwater_2019.nc"
    path_rh5_test = "predictors/test/rh500_2019.nc"
    path_rh3_test = "predictors/test/rh300_update_2019.nc"
    path_d2m_test = "predictors/test/d2m_2019.nc"
    path_cin_test = "predictors/test/cin_2019.nc"
    path_sh600_test = "predictors/test/spec_humid600_2019.nc"
    path_sh925_test = "predictors/test/spec_humid925_2019.nc"
    path_temp_test = "predictors/test/t2m_2019.nc"
    path_kindx_test = "predictors/test/kindx_2019.nc"
    path_sh7_test = "predictors/test/spec_humid700_2019.nc"
    path_shear_test = "predictors/test/shear925_600_2019.nc"
    path_stream_test = "predictors/test/stream_2019.nc"
    path_vimd_test = "predictors/test/vimd_2019.nc"
    path_pressure_tendency_test = "predictors/test/pressure_tendency_2019.nc"
    path_precip_lag1_test = "predictors/test/precip_obs_lag_1_sel_2019.nc"
    path_precip_lag2_test = "predictors/test/precip_obs_lag_2_sel_2019.nc"
    path_precip_lag3_test = "predictors/test/precip_obs_lag_3_sel_2019.nc"
    path_t850_test = "predictors/test/temp_850_2019.nc"
    path_t500_test = "predictors/test/temp_500_2019.nc"
    path_sh500_test = "predictors/test/spec_humid500_2019.nc"


def select_data_subset(paths, version='v1', fold=0):
    if version in ["v1", "v1+time"]:
        return subset_v1(paths, fold)
    elif version in ["v2", "v2+time"]:
        return subset_v2(paths, fold)
    else:
        raise NotImplemented("Data subset version not implemented.")
    
        
# Version 1: baseline

def subset_v1(paths, fold):
    train = [
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_1lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_2lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_3lag.nc",
    ]
    test = [
        "corr_predictors/predictor_test_18_19_1lag.nc",
        "corr_predictors/predictor_test_18_19_2lag.nc",
        "corr_predictors/predictor_test_18_19_3lag.nc",
    ]

    return train, test

# Version 2: all features

def subset_v2(paths, fold):
    train = [
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_1lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_2lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_3lag.nc",
        paths.path_pw_train,
        paths.path_cape_train,
        paths.path_cc_train,
        paths.path_clwc_train,
        paths.path_rh5_train,
        paths.path_rh3_train,
        paths.path_d2m_train,
        paths.path_cin_train,
        paths.path_sh600_train,
        paths.path_sh925_train,
        paths.path_temp_train,
        paths.path_kindx_train,
        paths.path_sh7_train,
        paths.path_shear_train,
        paths.path_vimd_train,
        paths.path_stream_train,
        paths.path_pressure_tendency_train,
        paths.path_t850_train,
        paths.path_t500_train,
        paths.path_sh500_train,
    ]
    test = [
        "corr_predictors/predictor_test_18_19_1lag.nc",
        "corr_predictors/predictor_test_18_19_2lag.nc",
        "corr_predictors/predictor_test_18_19_3lag.nc",
        paths.path_pw_test,
        paths.path_cape_test,
        paths.path_cc_test,
        paths.path_clwc_test,
        paths.path_rh5_test,
        paths.path_rh3_test,
        paths.path_d2m_test,
        paths.path_cin_test,
        paths.path_sh600_test,
        paths.path_sh925_test,
        paths.path_temp_test,
        paths.path_kindx_test,
        paths.path_sh7_test,
        paths.path_shear_test,
        paths.path_vimd_test,
        paths.path_stream_test,
        paths.path_pressure_tendency_test,
        paths.path_t850_test,
        paths.path_t500_test,
        paths.path_sh500_test,
    ]
    return train, test





def load_and_concat(
        data_dir,
        fold,
        list_of_features,
        add_time = False,
        mode = "train"
    ):
        
    with open(str(data_dir + '/split_train_folds.pickle'), 'rb') as f:
        X = pickle.load(f)
    
    if fold == 8:
        cv_fold = X[8]
        
    else:
        timeseries_cv_split_manual = X[0:8]
        cv_fold = timeseries_cv_split_manual[fold]
    
        
    data_list = []
    for i, feature in enumerate(list_of_features):
        if mode == "val" and "corr_predictors" in feature:
            feature = feature.replace("train", "test")  
        if mode == "val" and "upstream_predictors" in feature:
            feature = feature.replace("train", "test")
        dataset = xr.open_dataset(
            data_dir  + '/' + feature
        )
            
        data_array = dataset[list(dataset.data_vars)[0]].values

        if "corr_predictors" in feature or "precip_obs_lag" in feature:
            data_array = np.log(data_array + 0.001)
                
        if not "corr_predictors" in feature and not "upstream_predictors" in feature:
            if mode == "train":
                #print(len(data_array))
                data_array = data_array[cv_fold[0]]
                
            elif mode == "val":
                data_array = data_array[cv_fold[1]]

        if add_time and i == 0:
            days_oty = pd.date_range(start=dataset.time.values[0], end=dataset.time.values[-1], freq="D").dayofyear.to_numpy() - 1  # type: ignore
            days_oty = np.tile(
                np.expand_dims(days_oty, (1, 2)),
                (1, data_array.shape[1], data_array.shape[2]),
            )
            time_encoding_1 = np.sin(2 * np.pi * days_oty / 365)
            time_encoding_2 = np.cos(2 * np.pi * days_oty / 365)

            if len(time_encoding_1) > len(data_array):
                if mode == "train":
                    cv_idx = 0
                else:
                    cv_idx = 1
                    
                time_encoding_1 = time_encoding_1[cv_fold[cv_idx]]
                time_encoding_2 = time_encoding_2[cv_fold[cv_idx]]

            data_list.append(time_encoding_1)
            data_list.append(time_encoding_2)
            
        #print(len(data_array))
        data_list.append(data_array)
    
    

    target_filename = (
            "obs_precip_train.nc" if mode in ["train", "val"] else "obs_precip_test.nc"
        )
    target = xr.open_dataset(data_dir +   "/observation/" + target_filename)
    target_array = target["precipitationCal"].values
    if mode == "train":
        target_array = target_array[cv_fold[0]]
    elif mode == "val":
        target_array = target_array[cv_fold[1]]
    
    #print(len(target_array))
    #print(len(data_list[0]))

    return np.stack(data_list, axis=1), target_array


class PerFeatureMeanStdScaler:
    def __init__(self, axis: int = 1) -> None:
        self.axis = axis
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> None:
        self.n_features = data.shape[self.axis]
        self.mean = np.zeros(self.n_features)
        self.std = np.zeros(self.n_features)

        for i in range(self.n_features):
            self.mean[i] = data[:, i].mean()
            self.std[i] = data[:, i].std()

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("You must call fit before transform or inverse transform.")
        new_data = np.zeros(data.shape)
        for i in range(self.n_features):
            if self.std[i] == 0:
                new_data[:, i] = data[:, i] - self.mean[i]
            else:
                new_data[:, i] = (data[:, i] - self.mean[i]) / self.std[i]

        return new_data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("You must call fit before transform or inverse transform.")
        new_data = np.zeros(data.shape)
        for i in range(self.n_features):
            if self.std[i] == 0:
                new_data[:, i] = data[:, i] + self.mean[i]
            else:
                new_data[:, i] = (data[:, i] * self.std[i]) + self.mean[i]

        return new_data
    


def sel_season_indx(season, year_dim):
    leap_year = 0
    if year_dim == 366:
        leap_year = 1
    if season == 'DJF':
        ix0 = 0
        ix1 = 90+leap_year
    elif season == 'MA':
        ix0 = 90+leap_year
        ix1 = ix0 + 61
    elif season == 'MJ':
        ix0 = 151 + leap_year
        ix1 = ix0 + 61
    elif season == 'JAS':
        ix0 = 212 + leap_year
        ix1 = ix0 + 92
    elif season == 'ON':
        ix0 = 304 + leap_year 
        ix1 = ix0 + 61
    else:
        raise ValueError('season not defined')
    
    return ix0, ix1



def load_obs_time(
        data_dir,
        fold,
        mode = "train"
    ):
        
    with open(str(data_dir + '/split_train_folds.pickle'), 'rb') as f:
        X = pickle.load(f)
    
    if fold == 8:
        cv_fold = X[8]
        
    else:
        timeseries_cv_split_manual = X[0:8]
        cv_fold = timeseries_cv_split_manual[fold]
    
    

    target_filename = (
            "obs_precip_train.nc" if mode in ["train", "val"] else "obs_precip_test.nc"
        )
    target = xr.open_dataset(data_dir +   "/observation/" + target_filename)
    target_array = target["precipitationCal"].values
    time_vals_array = target.time.values
    if mode == "train":
        target_array = target_array[cv_fold[0]]
        time_vals_array = time_vals_array[cv_fold[0]]
    elif mode == "val":
        target_array = target_array[cv_fold[1]]
        time_vals_array = time_vals_array[cv_fold[1]]

    time_vals_pd = pd.date_range(start=pd.to_datetime(time_vals_array[0]), end = pd.to_datetime(time_vals_array[-1]))
    time_vals_pd = time_vals_pd - pd.Timedelta(1, unit='D') 
    return target_array, time_vals_pd.month.to_numpy()


def load_obs(
        data_dir,
        fold,
        mode = "train"
    ):
        
    with open(str(data_dir + '/split_train_folds.pickle'), 'rb') as f:
        X = pickle.load(f)
    
    if fold == 8:
        cv_fold = X[8]
        
    else:
        timeseries_cv_split_manual = X[0:8]
        cv_fold = timeseries_cv_split_manual[fold]

    target_filename = (
            "obs_precip_train.nc" if mode in ["train", "val"] else "obs_precip_test.nc"
        )
    target = xr.open_dataset(data_dir +   "/observation/" + target_filename)
    target_array = target["precipitationCal"].values
    if mode == "train":
        target_array = target_array[cv_fold[0]]
    elif mode == "val":
        target_array = target_array[cv_fold[1]]

    return target_array



def hres_season_time(season, year):
    if season == 'DJF':
        date_test = pd.date_range(start='12/02/20'+str(year-1)+'T06', end = '03/01/20'+str(year)+'T06')
    elif season == 'JAS':
        date_test = pd.date_range(start='07/02/20'+str(year)+'T06', end = '10/01/20'+str(year)+'T06')
    else:
        raise ValueError('season not defined')
    
    date_train = pd.date_range(start='12/02/2001T06', end ='12/01/20'+str(year-1)+'T06')
    return date_train, date_test


def ecmwf_season_time(season, year):
    if season == 'DJF':
        date_test = pd.date_range(start='12/02/20'+str(year-1)+'T06', end = '03/01/20'+str(year)+'T06')
    elif season == 'JAS':
        date_test = pd.date_range(start='07/02/20'+str(year)+'T06', end = '10/01/20'+str(year)+'T06')
    else:
        raise ValueError('season not defined')
    
    date_train = pd.date_range(start='12/02/2006T06', end ='12/01/20'+str(year-1)+'T06')
    return date_train, date_test




def CRPS_emosgev0 (obs,hres, ctrl, prtb, result_dic, p0, ens_gini):
    eps = 1e-5 
    nObs = obs.size
    A = result_dic['a'][0]
    C = result_dic['c'][0]
    D = result_dic['d'][0]
    Q = result_dic['q'][0]
    S = result_dic['s'][0]
    B = result_dic['B']
    x = np.insert(result_dic['B'],0,result_dic['a'])
    if nObs == 1:
        h_and_c = np.array([1, hres, ctrl])
        Af = np.concatenate((h_and_c, prtb))
        MEAN = np.sum(Af*x)+S*p0
        SCALE = C + D*ens_gini
        #print(SCALE)
        if np.absolute(Q) < eps:
            crps_eps_m = crps_GEVneq0(MEAN,SCALE,-eps, obs)
            crps_eps_p = crps_GEVneq0(MEAN,SCALE,eps, obs)
            w_m = (eps-Q)/(2*eps)
            w_p = (eps+Q)/(2*eps)
            return w_m*crps_eps_m + w_p*crps_eps_p
        else:
            #print(MEAN -SCALE*(gamma(1+Q)-1)/(-1*Q))
            ##print(SCALE)
            #print(Q)
            return crps_GEVneq0(MEAN, SCALE, Q, obs)
            
    else:
        h_and_c = np.array([np.ones(nObs), hres, ctrl])
        Af = np.concatenate((h_and_c.transpose(), prtb), axis=1)
        MEAN = Af.dot(x) + S*p0
        
        SCALE = np.repeat(result_dic['c'], nObs) + np.repeat(result_dic['c'],nObs)*ens_gini
        
        crps = np.zeros(nObs)
        for i in range(nObs):
            if np.absolute(Q) < eps:
                crps_eps_m = crps_GEVneq0(MEAN[i],SCALE[i],-eps, obs[i])
                crps_eps_p = crps_GEVneq0(MEAN[i],SCALE[i],eps, obs[i])
                w_m = (eps-Q)/(2*eps)
                w_p = (eps+Q)/(2*eps)
                crps[i] = w_m*crps_eps_m + w_p*crps_eps_p
            else:
                crps[i] = crps_GEVneq0(MEAN[i], SCALE[i], Q, obs[i])
        return crps

        
def crps_GEVneq0(mean, scale, shape, obs):
    loc = mean -scale*(gamma(1-shape)-1)/shape
    #print(loc)
    SCdSH = scale/shape
    Gam1mSH = gamma(1-shape)
    prob0 = genextreme.cdf(0, loc=loc, scale=scale, c=-1*shape)
    probY = genextreme.cdf(obs, loc=loc, scale=scale, c=-1*shape)
    
    T1 = (obs-loc)*(2*probY-1) + loc*prob0**2
    if prob0 == 0:
        T2 =SCdSH * ( 1-prob0**2 - 2**shape*Gam1mSH*1)
    else:
        T2 = SCdSH * ( 1-prob0**2 - 2**shape*Gam1mSH*stats.gamma.cdf(-2*np.log(prob0),1-shape))
    #pgamma(-2*log(prob0),1-SHAPE) )
    if probY == 0:
        T3 = -2*SCdSH * ( 1-probY - Gam1mSH*1)
    else:
        T3 = -2*SCdSH * ( 1-probY - Gam1mSH*stats.gamma.cdf(-np.log(probY),1-shape))
    return( np.mean(T1+T2+T3) )

def spear_season_ix(season):
    yearly_times = pd.date_range(start='12/02/2000T06', end='12/01/2019T06')
    if season == 'JAS':
        train_t = pd.date_range(start='07/02/2001T06', end='10/01/2001T06')
        for i in range(2, 20):
            if i < 10:
                t0 = pd.date_range(start='07/02/200' + str(i) + 'T06', end='10/01/200' + str(i) + 'T06')
            else:
                t0 = pd.date_range(start='07/02/20' + str(i) + 'T06', end='10/01/20' + str(i) + 'T06')
            train_t = train_t.union(t0)
    
    elif season == 'MA':
        train_t = pd.date_range(start='03/02/2001T06', end='05/01/2001T06')
        for i in range(2, 20):
            if i < 10:
                t0 = pd.date_range(start='03/02/200' + str(i) + 'T06', end='05/01/200' + str(i) + 'T06')
            else:
                t0 = pd.date_range(start='03/02/20' + str(i) + 'T06', end='05/01/20' + str(i) + 'T06')
            train_t = train_t.union(t0)   
    
    elif season == 'MJ':
        train_t = pd.date_range(start='05/02/2001T06', end='07/01/2001T06')
        for i in range(2, 20):
            if i < 10:
                t0 = pd.date_range(start='05/02/200' + str(i) + 'T06', end='07/01/200' + str(i) + 'T06')
            else:
                t0 = pd.date_range(start='05/02/20' + str(i) + 'T06', end='07/01/20' + str(i) + 'T06')
            train_t = train_t.union(t0)
    elif season == 'ON':
        train_t = pd.date_range(start='10/02/2001T06', end='12/01/2001T06')
        for i in range(2, 20):
            if i < 10:
                t0 = pd.date_range(start='10/02/200' + str(i) + 'T06', end='12/01/200' + str(i) + 'T06')
            else:
                t0 = pd.date_range(start='10/02/20' + str(i) + 'T06', end='12/01/20' + str(i) + 'T06')
            train_t = train_t.union(t0)

    elif season == 'DJF':
        train_t = pd.date_range(start='12/02/2000T06', end='03/01/2001T06')
        for i in range(2, 20):
            if i < 10:
                t0 = pd.date_range(start='12/02/200' + str(i - 1) + 'T06', end='03/01/200' + str(i) + 'T06')
            else:
                if i == 10:
                    t0 = pd.date_range(start='12/02/200' + str(i - 1) + 'T06', end='03/01/20' + str(i) + 'T06')
                else:
                    t0 = pd.date_range(start='12/02/20' + str(i - 1) + 'T06', end='03/01/20' + str(i) + 'T06')
            train_t = train_t.union(t0)
    else:
        raise ValueError('season not defined')
    indices = np.where(np.in1d(yearly_times, train_t))[0]
    return indices

def spear_corr(season, data_dir):
    lsm = np.loadtxt(data_dir  + "/lsm.txt")
    land_bin = np.where(lsm == 0, np.nan, lsm)
    lons = np.arange(-25, 35.5)
    lats = np.arange(19)
    lon_flat = np.tile(lons, 19)
    lat_flat = np.repeat(lats, 61)

    combi = np.vstack([lat_flat, lon_flat, land_bin.flatten()]).T
    combi_df = pd.DataFrame(combi)
    combi_df_nan = combi_df.dropna()
    lat_tuple = np.asarray(combi_df_nan[0])
    lon_tuple = np.asarray(combi_df_nan[1])
    
    paths = PrecipitationDataPaths()
    feature_set_train, feature_set_test = select_data_subset(paths=paths, version='v2', fold=8)
    Xtrain, ytrain = load_and_concat(data_dir, 8, feature_set_train, add_time = False, mode = "train")
    Xtest, ytest = load_and_concat(data_dir, 8, feature_set_train, add_time = False, mode = "test")
    M = Xtrain.shape[1]
    spear_vals = np.zeros((len(lat_tuple), M, M))
    indices = spear_season_ix(season)
    for i in range(len(lat_tuple)):
        lat = int(lat_tuple[i])
        lon = lon_tuple[i]
        j = int(lon + 25)
        Xtrain_grid = Xtrain[:, :, lat, j]
        Xtest_grid = Xtest[:, :, lat, j]
        X_grid = np.concatenate((Xtrain_grid, Xtest_grid))
        r = spearmanr(X_grid[indices,:], axis = 0)
        spear_vals[i, :, :] = r[0]
    return spear_vals


def cpa(response, predictor):
    """
    Calculate CPA coefficient.

    CPA attains values between zero and one. Weighted probability of concordance. 

    Parameters
    ----------
    response : 1D array_like, 1-D array containing observation (response). Need to have the same length in the ``axis`` dimension as predictor.
    predictor : 1D array_like, 1-D array containing predictions for observation.
       
    Returns
    -------
    correlation : float
        	  CPA coefficient 
    """    
    response = np.asarray(response)
    if response.ndim > 1:
        raise ValueError("CPA only handles 1-D arrays of responses")

    predictor = np.asarray(predictor)
	
    if predictor.ndim > 1:
        ValueError("CPA only handles 1-D arrays of forecasts")   
  
    	# check for nans
    if np.isnan(np.sum(response)) == True:
        ValueError("response contains nan values")
		
    if np.isnan(np.sum(predictor)) == True:
        ValueError("forecast contains nan values")
	
    #responseOrder = np.argsort(response)
    #responseSort = response[responseOrder] 
    #forecastSort = predictor[responseOrder]                
    forecastRank = rankdata(predictor, method='average')
    responseRank = rankdata(response, method='average')
    responseClass = rankdata(response, method='dense')
    
    return((np.cov(responseClass,forecastRank)[0][1]/np.cov(responseClass,responseRank)[0][1]+1)/2) 
	
 


def cpa_seasons(data_dir, accum = True):

    lsm = np.loadtxt(data_dir  + "/lsm.txt")
    land_bin = np.where(lsm == 0, np.nan, lsm)
    lons = np.arange(-25, 35.5)
    lats = np.arange(19)
    lon_flat = np.tile(lons, 19)
    lat_flat = np.repeat(lats, 61)

    combi = np.vstack([lat_flat, lon_flat, land_bin.flatten()]).T
    combi_df = pd.DataFrame(combi)
    combi_df_nan = combi_df.dropna()
    lat_tuple = np.asarray(combi_df_nan[0])
    lon_tuple = np.asarray(combi_df_nan[1])
    
    paths = PrecipitationDataPaths()
    feature_set_train, feature_set_test = select_data_subset(paths=paths, version='v2', fold=8)
    Xtrain, ytrain = load_and_concat(data_dir, 8, feature_set_train, add_time = False, mode = "train")
    Xtest, ytest = load_and_concat(data_dir, 8, feature_set_train, add_time = False, mode = "test")
    M = Xtrain.shape[1]

    col_gradient = plt.get_cmap('Blues', M)
    newcolors = col_gradient(np.linspace(0, 1, M))
    newcmp = ListedColormap(newcolors)
    colors = [mcolors.to_hex(newcolors[i,:]) for i in np.arange((M-1), -1, -1)]

    names = ['corr1', 'corr2', 'corr3', 'TCWV', 'CAPE', 'TCC', 'TCLW', 'R500', 'R300', 'D2', 'CIN', 'Q600', 
         'Q925', 'T2', 'KX','Q700', 'SHR',  'VIMD', r'$\Psi$700','SPT', 'T850', 'T500', 'Q500']

    if accum:
        ix = np.array([3, 15, 14, 12, 9, 6, 22, 11, 7, 4, 5, 8, 10, 13, 16, 19, 21, 20, 17, 18])
    else:
        ix = np.array([3, 15, 14, 12, 9, 6, 22, 11, 7, 4, 5, 8, 10, 13, 16, 19, 21, 20, 17, 18])

    names_ix_list = [names, ix]

    df_list = list()
    mean_list = list()
    color_list = list()
    ranking_list = list()
    
    for season in ['DJF', 'MA', 'MJ', 'JAS', 'ON']:
        indices = spear_season_ix(season)
        cpa_vals = np.zeros((len(lat_tuple), M))
        for i in range(len(lat_tuple)):
            #print(i)
            lat = int(lat_tuple[i])
            lon = lon_tuple[i]
            j = int(lon + 25)
            Xtrain_grid = Xtrain[:, :, lat, j]
            Xtest_grid = Xtest[:, :, lat, j]
            ytrain_grid = ytrain[:, lat, j]
            ytest_grid = ytest[:, lat, j]
            y_grid = np.concatenate((ytrain_grid, ytest_grid))
            if not accum:
                y_grid = y_grid > 0.2
            X_grid = np.concatenate((Xtrain_grid, Xtest_grid))
            if not accum and np.sum(y_grid[indices]) == 0:
                cpa_vals[i, :] = np.nan
            else:
                for k in range(M):
                    cpa_vals[i, k] = cpa(y_grid[indices], X_grid[indices, k])
        cpa_mean = np.nanmean(cpa_vals[:, ix], axis = 0)
        df = pd.DataFrame(cpa_vals[:, ix], columns = np.asarray(names)[ix])
        df = df.dropna()
        df_list.append(df)
        mean_list.append(cpa_mean)

        # color coding
        ix_cpa = np.argsort(cpa_mean)[::-1]
        rem_colors = np.asarray(colors)
        ranking_ordered = np.arange(len(rem_colors))
        k = 0
        for ixx in ix_cpa:
            rem_colors[ixx] = colors[k]
            ranking_ordered[ixx] = k
            k = k+1

        ranking_list.append(ranking_ordered)
        color_list.append(rem_colors)

    return df_list, color_list, ranking_list, mean_list, names_ix_list


def label_panel(ax, letter, *,offset_left=0.0, offset_up=0.2, prefix='', postfix=')', **font_kwds):
    kwds = dict(fontsize=26)
    kwds.update(font_kwds)
    # this mad looking bit of code says that we should put the code offset a certain distance in
    # inches (using the fig.dpi_scale_trans transformation) from the top left of the frame
    # (which is (0, 1) in ax.transAxes transformation space)
    fig = ax.figure
    trans = ax.transAxes + transforms.ScaledTranslation(-offset_left, offset_up, fig.dpi_scale_trans)
    ax.text(0, 1, prefix+letter+postfix, transform=trans, **kwds)
    
def label_panels(axes, letters=None, **kwds):
    if letters is None:
        letters = axes.keys()
    for letter in letters:
        ax = axes[letter]
        label_panel(ax, letter, **kwds)
