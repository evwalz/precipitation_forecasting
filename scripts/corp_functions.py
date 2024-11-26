from scipy.interpolate import interp1d
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr
import random


class reldiag:
    def __init__(self, cases, bins, regions, scores, ptype):
        self.cases = cases
        self.bins = bins
        self.regions = regions
        self.scores = scores
        self.ptype = ptype

    def corp_plot(self):
        fig, ax = plt.subplots()
        cali_fct = self.cases['CEP_pav'].to_numpy()
        probs_grid_test = self.cases['x'].to_numpy()
        ax.plot([0, 1], [0, 1], 'black', linewidth=0.5)
        data = self.cases['x'].to_numpy()
        if self.ptype == 'discrete':
            x_unique = np.sort(np.unique(data))
            eps = np.minimum(np.min(np.diff(x_unique)) / 8, 0.02)
            bin_seq = np.repeat(x_unique, 2) + np.tile(np.array([-eps, eps]), len(x_unique))
            weights = np.ones_like(data) / len(data)
            ax.hist(data, weights=weights, facecolor='None', ec='black', bins=bin_seq)

            ax.plot(np.sort(probs_grid_test), np.sort(cali_fct), color='red', linewidth=0.5)
            ax.plot(np.sort(probs_grid_test), np.sort(cali_fct),'.', color='red')
        else:
            iqr_val = iqr(data)
            if iqr_val < 0.000001:
                bin_size = 1 / 400
            else:
                bin_size = 2 * iqr_val / len(data) ** (1 / 3)
            val = int(np.round((np.max(data) - np.min(data)) / bin_size))
            if val < 5:
                val = 5
            bin_seq = np.linspace(start=np.min(data), stop=np.max(data), num=(val + 1))
            weights = np.ones_like(data) / len(data)
            ax.hist(data, weights=weights, facecolor='None', ec='black', bins=bin_seq)

            ax.plot(np.sort(probs_grid_test), np.sort(cali_fct), color='red', linewidth=0.5)
            bmin = self.bins['x_min'].to_numpy()
            bmax = self.bins['x_max'].to_numpy()
            bcali = self.bins['CEP_pav'].to_numpy()
            for k in range(len(bcali)):
                seg = [bmin[k], bmax[k]]
                y = bcali[k]
                ax.plot(seg, [y, y], color='red', linewidth=3)
            # plt.plot(probs_grid_test, cali_fct, '.', color = 'red')

        ax.set_xlabel('Forecast probability')
        # ax[i].set_ylabel('Conditional event probability')
        ax.set_ylabel('Conditional event probability')
        ax.fill_between(self.regions['x'].to_numpy(), self.regions['lower'].to_numpy(), self.regions['upper'].to_numpy(), alpha=.15,
                            color='#0000FF')

        mcb = str(np.round(self.scores['mcb'][0], 3))
        dsc = str(np.round(self.scores['dsc'][0], 3))
        unc = str(np.round(self.scores['unc'][0], 3))
        ax.text( 0.01, 0.95, 'MCB = ' +mcb , fontsize = 12)
        ax.text( 0.01, 0.88, 'DSC = '+dsc, fontsize = 12)
        ax.text( 0.01, 0.81, 'UNC = '+unc, fontsize = 12) 

    
def reliabilitydiag(p, ybin, region_position = 'diagonal'):
    if any(p > 1) or any(p < 0):
        raise ValueError("p must be between 0 and 1")
    y_disc = np.sort(np.unique(ybin))
    if len(y_disc) == 1:
        raise ValueError("y must contain 0 and 1")
    if y_disc[0] != 0 or y_disc[1] != 1:
        raise ValueError("y must only contain 0 and 1")
    if len(p) != len(ybin):
        raise ValueError("p and y must have same length")

    ptype = detect_ptype(p)
    x = p.copy()
    y = ybin.copy()
    x_order = np.argsort(x)
    x = x[x_order]
    y = y[x_order]
    pav_x = CEP_pav(x, y)

    # cases:
    cases = pd.DataFrame({'x': x, 'CEP_pav': pav_x})  # np.transpose(np.vstack([x, pav_x]))

    # bins:
    bins_cep_pav, bins_len = find_runs(pav_x)
    red_iKnots = np.cumsum(bins_len)
    bins_min = x[np.insert(red_iKnots[0:-1], 0, 0)]
    bins_max = x[(red_iKnots - 1)]
    bins = pd.DataFrame({'x_min': bins_min, 'x_max': bins_max,
                         'CEP_pav': bins_cep_pav})  # np.transpose(np.vstack([bins_cep_pav, bins_min, bins_max]))

    # scores:
    mean_score = np.mean(brier(y, x))
    uncertainty = np.mean(brier(y, np.mean(y)))
    Sc = np.mean(brier(y, pav_x))
    discrimination = uncertainty - Sc
    miscalibration = mean_score - Sc
    scores = pd.DataFrame({'ms':[mean_score], 'mcb':[miscalibration], 'dsc': [discrimination], 'unc':[uncertainty]})

    # region_method equals resampling
    regions = region_resampling(cases, bins, region_level=0.9, region_position=region_position, n_boot=100)
    reldiag_object = reldiag(cases = cases, bins = bins, regions = regions, scores = scores, ptype = ptype)
    return reldiag_object


def brier(y, x):
    return (x - y) ** 2


# uses ties handling "secondary"
def CEP_pav(x, y):
    return IsotonicRegression().fit_transform(x, y)


def decomposition(forecast, obs):
    x = forecast.copy()
    y = obs.copy()
    x_order = np.argsort(x)
    x = x[x_order]
    y = y[x_order]
    mean_score = np.mean(brier(y, x))
    uncertainty = np.mean(brier(y, np.mean(y)))
    Sc = np.mean(brier(y, CEP_pav(y, x)))
    discrimination = uncertainty - Sc
    miscalibration = mean_score - Sc
    return mean_score, miscalibration, discrimination, uncertainty


def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_lengths  # mrun_starts,



def region_resampling(df_pav, df_bins, region_level, region_position, n_boot):
    regions = df_pav.groupby('x').agg(CEP_pav=('CEP_pav', lambda x: list(np.unique(x))),  # unique values of CEP_pav
                                      n=('x', 'count')).reset_index()
    n_pav = df_pav.shape[0]
    n_regions = regions.shape[0]
    if region_position == 'estimate':
        x0 = df_pav['CEP_pav']
    else:
        x0 = df_pav['x']

    def isofit(y):
        return IsotonicRegression().fit_transform(np.arange(len(y)), y)

    boot_samples = []
    for _ in range(n_boot):
        s = random.choices(np.arange(n_pav), k=n_pav)
        x = df_pav['x'].to_numpy()[s]
        # n, size, p
        y = np.random.binomial(1, x0.to_numpy()[s], n_pav)

        x_order = np.argsort(x)
        x = x[x_order]
        y = y[x_order]
        res_fit = pd.DataFrame({'x': x, 'CEP_pav': CEP_pav(x, y)})
        res_fit = res_fit.drop_duplicates()
        if res_fit.shape[0] == 1:
            boot_samples.append(res_fit['CEP_pav'])
        else:
            merged_df = pd.merge(regions, res_fit, on='x', how='left')
            sub_merged_df = merged_df[['x', 'CEP_pav_y']].dropna()

            approx_nan = np.interp(merged_df['x'].to_numpy(), sub_merged_df['x'].to_numpy(),
                                   sub_merged_df['CEP_pav_y'].to_numpy())
            CEP_pav_y = merged_df['CEP_pav_y'].to_numpy()
            ix = np.isnan(CEP_pav_y)
            CEP_pav_y[ix] = approx_nan[ix]
            boot_samples.append(CEP_pav_y)
        # merged_df['CEP_pav_y'] = CEP_pav_y

    boot_samples_mat = np.transpose(np.array(boot_samples))
    probs = 0.5 + np.array([-0.5, 0.5]) * region_level
    bounds = np.apply_along_axis(np.quantile, axis=1, arr=boot_samples_mat,
                                 q=probs)

    def b_corr(b):
        return bound_correction(b, regions['x'], regions['CEP_pav'], region_position)

    lbound = b_corr(bounds[:, 0])
    ubound = b_corr(bounds[:, 1])

    return pd.DataFrame(
        {'x': regions['x'], 'lower': lbound, 'upper': ubound})  # np.transpose(np.vstack([regions_x, lbound, ubound]))


def bound_correction(bound, x, CEP_est, position):
    # Since position always diagonal, don't need "else" for now: might contain errors
    if len(x) == 1 or position == "diagonal":
        return bound
    else:
        # Conditionally apply operations using numpy
        condition = np.logical_and(CEP_est.isin([0, 1]), ~np.isin(x, np.arange(x.min(), x.max() + 1)))
        result = np.where(condition, np.nan, bound)
        # Use scipy's interp1d for interpolation
        interp_func = interp1d(x, result, fill_value="extrapolate")
        result_y = interp_func(x)

        # Access the interpolated values
        return result_y


def detect_ptype(x):
    x_unique = np.sort(np.unique(x))
    if len(x_unique) == 1:
        return 'discrete'
    elif np.min(np.diff(x_unique)) >= 0.01:
        return 'discrete'
    else:
        return 'continuous'
        




