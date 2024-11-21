# Replication for "24-Hour Probabilistic Forecasts of Precipitation for Northern Tropical Africa "

### Content 
Code to build data-driven models, to produce precipitation forecasts and to replicate Figure 9 from the paper "Physics-Based vs Data-Driven 24-Hour Probabilistic Forecasts of Precipitation for Northern Tropical Africa".


The folder [scipts](./scipts/) contains jupyter notebooks to compute data-driven forecasts, an R-file to compute EMOS method and a jupyter notebook to visualize results (Figure 9 of the paper).

The folder [precip_data](./precip_data/) contains data to compute climatology, statistical forecasts (DIM and Logit) and to replicate Figure 9. To compute scores for NWP foreacsts and the hybrid models NWP data in the folder [forecasts](./precip_data/forecasts) is required which is not provided in this repository (see next section)

The folder [forecasts](./precip_data/forecasts) is empty since data size is to large. Contact us to obtain access to the CNN forecast data, the high resolution (HRES) run and the ensemble data.

### Instruction

File paths are realtive to root of this package. If necessary adapt path to data in the files of the folder [scipts](./scipts/). 

Notebook `plot_eval.ipynb` computes Figure 9.

To compute CNN forecasts follow instructions under [https://github.com/evwalz/precipitation](https://github.com/evwalz/precipitation)
Data to compute CNN forecasts is provided in folder [precip_data](https://github.com/evwalz/precipitation_forecasting/precip_data). Set path to data directory in script [run.sh](https://github.com/evwalz/precipitation/run/).

### References
Walz, E., P. Knippertz, A. H. Fink, G. Köhler, and T. Gneiting, 2024: Physics-Based vs Data-Driven 24-Hour Probabilistic Forecasts of Precipitation for Northern Tropical Africa. Mon. Wea. Rev., 152, 2011–2031, [https://doi.org/10.1175/MWR-D-24-0005.1](https://doi.org/10.1175/MWR-D-24-0005.1). 

