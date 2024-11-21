from dataclasses import dataclass
import pickle
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import genextreme
from scipy.special import gamma 
from scipy import stats  


@dataclass
class PrecipitationDataPaths:
    """Class to handle precipitation data."""

    path_pw_train = "predictors/train/tcwv_2000_2018.nc"
    path_cape_train = "predictors/train/cape_2000_2018.nc"
    path_cc_train = "predictors/train/cloudcover_2000_2018.nc"
    path_clwc_train = "predictors/train/cloudwater_2000_2018.nc"
    path_rh5_train = "predictors/train/rh500_2000_2018.nc"
    path_rh3_train = "predictors/train/rh300_update_2000_2018.nc"
    path_d2m_train = "predictors/train/d2m_2000_2018.nc"
    path_cin_train = "predictors/train/cin_2000_2018.nc"
    path_vo7_train = "predictors/train/relvor700_2000_2018.nc"
    path_sh600_train = "predictors/train/spec_humid600_2000_2018.nc"
    path_sh925_train= "predictors/train/spec_humid925_2000_2018.nc"
    path_temp_train = "predictors/train/t2m_2000_2018.nc"
    path_kindx_train = "predictors/train/kindx_2000_2018.nc"
    path_sh7_train = "predictors/train/spec_humid700_2000_2018.nc"
    path_sp_train = "predictors/train/surfpressure_2000_2018.nc"
    path_shear_train = "predictors/train/shear925_600_2000_2018.nc"
    path_stream_train = "predictors/train/stream_2000_2018.nc"
    path_geodiff_train = "predictors/train/geodiff_2000_2018.nc"
    path_vertvelo_train = "predictors/train/vert_velocity_mean850_500_300_2000_2018.nc"
    path_vimd_train = "predictors/train/accum_vimd_2000_2018.nc"
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
    path_vo7_test = "predictors/test/relvor700_2019.nc"
    path_sh600_test = "predictors/test/spec_humid600_2019.nc"
    path_sh925_test = "predictors/test/spec_humid925_2019.nc"
    path_temp_test = "predictors/test/t2m_2019.nc"
    path_kindx_test = "predictors/test/kindx_2019.nc"
    path_sh7_test = "predictors/test/spec_humid700_2019.nc"
    path_sp_test = "predictors/test/surfpressure_2019.nc"
    path_shear_test = "predictors/test/shear925_600_2019.nc"
    path_stream_test = "predictors/test/stream_2019.nc"
    path_geodiff_test = "predictors/test/geodiff_2019.nc"
    path_vertvelo_test = "predictors/test/vert_velocity_mean850_500_300_2019.nc"
    path_vimd_test = "predictors/test/accum_vimd_2019.nc"
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

