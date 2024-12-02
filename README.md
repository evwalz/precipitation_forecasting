# Replication for "24-Hour Probabilistic Forecasts of Precipitation for Northern Tropical Africa "

### Content and Instruction
Code to build data-driven models, to produce precipitation forecasts and to replicate Figure 9 from the paper "Physics-Based vs Data-Driven 24-Hour Probabilistic Forecasts of Precipitation for Northern Tropical Africa".

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
 
To compute data to reproduce Figure 10 and 11 use same scripts but set `save_full = True`. Same scripts can be used to obtain forecast data to compute BS and CRPS decomposition. Decomposition components to reproduce Figure 14 are provided in folder [results](./precip_data/results/). (To compute CRPS decomposition components use function `isodeco_crps` from R package [isodisregSD](https://github.com/evwalz/isodisregSD) and for BS decomposition use file `deco_bs.R`)

The folder [precip_data](./precip_data/) contains data to compute climatology, statistical forecasts (DIM and Logit) and to replicate Figure 9. To compute scores for NWP foreacsts and the hybrid models NWP data in the folder [forecasts](./precip_data/forecasts) is required which is not provided in this repository since data size is too large. Contact us to obtain access to the CNN forecast data, the high resolution (HRES) run and the ensemble data.

To compute CNN forecasts follow instructions under [https://github.com/evwalz/precipitation](https://github.com/evwalz/precipitation). Data to compute CNN forecasts is provided in folder [precip_data](./precip_data/). Set path to data directory in script [run.sh](https://github.com/evwalz/precipitation/tree/main/run).

### Data
To run the code several data sources are required:

    1. [GPM IMERG](https://gpm.nasa.gov/data/imerg) data in folder [observation](./precip_data/observation). Data is processed as described [here](https://github.com/evwalz/epc). The referenced paper describes how to compute the correlated predictors
    2. Get [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) data and save it in folder [predictors](./precip_data/predictors). The referenced paper describes how to preprocess raw data.
    3. Get ECMWF HRES and ensemble data from [MARS](https://confluence.ecmwf.int/display/CEMS/MARS) archive and save it in folder [forecasts](./precip_data/forecasts). The referenced paper describes how to preprocess raw data.


### References
Walz, E., P. Knippertz, A. H. Fink, G. Köhler, and T. Gneiting, 2024: Physics-Based vs Data-Driven 24-Hour Probabilistic Forecasts of Precipitation for Northern Tropical Africa. Mon. Wea. Rev., 152, 2011–2031, [https://doi.org/10.1175/MWR-D-24-0005.1](https://doi.org/10.1175/MWR-D-24-0005.1). 

Huffman, G. J., and Coauthors, 2020: Integrated multi-satellite retrievals for the Global Precipitation Measurement (GPM) mission (IMERG). Satellite Precipitation Measurement:
Volume 1, V. Levizzani et al., Eds., Springer, 343–353.



