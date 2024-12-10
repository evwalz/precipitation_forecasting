# Replication for "24-Hour Probabilistic Forecasts of Precipitation for Northern Tropical Africa "

## Content and Instruction
Code to build data-driven models, to produce precipitation forecasts and to replicate Figures from the paper "Physics-Based vs Data-Driven 24-Hour Probabilistic Forecasts of Precipitation for Northern Tropical Africa".

File paths are relative to folder structure in this package. If necessary adapt path to data in the files of the folder [scripts](./scripts/). The folder [scripts](./scripts/) contains jupyter notebooks to compute data-driven forecasts, an R-file to compute EMOS method and jupyter notebooks to visualize results:
    
    - Figure 2, 3 and 4: cpa.ipynb 
    - Figure 5: spear_correlation.ipynb 
    - Figure 9: plot_eval.ipynb
    - Figure 10 and 11: skillscore.ipynb
    - Figure 12 and 13: Niamey.ipynb
    - Figure 14: decomposition.ipynb

For Figure 7 and 8 successively remove one feature to train logit and dim model (stat_models.ipynb). To replicate data which is used for Figure 9 (see folder [results](./precip_data/results/)) run forecast scripts in folder [fct_models](./scripts/fct_models/): 

    - mpc.ipynb
    - stat_models.ipynb
    - cnn_model.ipynb
    - ecmwf_fct.ipynb 
    - emos.R
 
To compute data to reproduce Figure 10 and 11 use same scripts but set `save_full = True`. Same scripts can be used to obtain forecast data to compute BS and CRPS decomposition. Decomposition components to reproduce Figure 14 are provided in folder [results](./precip_data/results/). (To compute CRPS decomposition components use function `isodeco_crps` from R package [isodisregSD](https://github.com/evwalz/isodisregSD) and for BS decomposition use file `deco_bs.R` in folder [scripts](./scripts/))

## Data
To run the code several data sources are required and should be stored in folder [precip_data](./precip_data/). In this repository, we provide the following data:

#### GPM IMERG

 [GPM IMERG](https://gpm.nasa.gov/data/imerg) data in folder [observation](./precip_data/observation). Data is processed as described [here](https://github.com/evwalz/epc). The referenced paper describes how to compute the correlated predictors which are provided in [corr_predictors](./precip_data/corr_predictors).

#### CNN
CNN forecasts in folder [cnn_fct](./precip_data/forecasts/cnn_fct). To compute CNN forecasts follow instructions under [https://github.com/evwalz/precipitation](https://github.com/evwalz/precipitation). Data to compute CNN forecasts is provided in folder [precip_data](./precip_data/). Set path to data directory in script [run.sh](https://github.com/evwalz/precipitation/tree/main/run).

#### ERA5

[ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) reanalysis data.
Use scripts in folder [download_data](./scripts/download_data/) to download (`era5_download.ipynb`) and preprocess data (`era5_preprocess.ipynb`). The final output is saved in [predictors](./precip_data/predictors). The download scripts require to set up climate data store (CDS) API. Alternatively, data can be downloaded through the CDS web interface. 

#### ECMWF HRES
Get ECMWF HRES data and save it in folder [hres_fct](./precip_data/forecasts/hres_fct). 
The referenced paper describes how to preprocess raw data. Adapt filename of data in scripts. 

#### ECMWF ensemble
Get ECMWF ensemble data and save it in folder [ensemble_fct](./precip_data/forecasts/ensemble_fct).
The referenced paper describes how to preprocess raw data. Construct one file with probability of precipitation (PoP) forecasts from 2006 to 2019 (adapt filename in code). Construct 9 files with the 51-member ensemble data for each year from 2011 to 2019, called `ens_0.nc` up to `ens_8.nc`. 
To compute emos, download and save HRES, CTRL and PRTB forecast data in folder (emos)[./precip_data/forecasts/ensemble_fct/emos]. In script `emos.R`, data is called `hres_2006.nc`, `ctrl_2006.nc` and `prtb_2006.nc`. 

<!--from [MARS](https://confluence.ecmwf.int/display/CEMS/MARS) archive -->

## Work in progress

Include computation of stream function.


## References
Walz, E., P. Knippertz, A. H. Fink, G. Köhler, and T. Gneiting, 2024: Physics-Based vs Data-Driven 24-Hour Probabilistic Forecasts of Precipitation for Northern Tropical Africa. Mon. Wea. Rev., 152, 2011–2031, [https://doi.org/10.1175/MWR-D-24-0005.1](https://doi.org/10.1175/MWR-D-24-0005.1). 

Huffman, G. J., and Coauthors, 2020: Integrated multi-satellite retrievals for the Global Precipitation Measurement (GPM) mission (IMERG). Satellite Precipitation Measurement:
Volume 1, V. Levizzani et al., Eds., Springer, 343–353.



