
library(scoringRules)
library(ncdf4)
library(isodistrreg)
library(ensembleMOS)

#source('emos_functions.R')
source(paste(dirname(rstudioapi::getSourceEditorContext()$path),'/emos_functions.R', sep = ''))


data_dir <- "../precip_data"
data_dir <- "/Volumes/My Passport for Mac/cnn/data_update/with_precip/precip_data"

lsm = read.table(paste(data_dir,'/lsm.txt' , sep = ''))
colnames(lsm) <- NULL
rownames(lsm) <- NULL
lsm_bin <- as.matrix(lsm)

ix <- which(lsm_bin == 0, arr.ind = T)
lsm_bin[ix] <- NaN


prtb <- nc_open(paste(data_dir,'/forecasts/ensemble_fct/emos/prtb_2006.nc', sep = ''))
hres <- nc_open(paste(data_dir,'/forecasts/ensemble_fct/emos/hres_2006.nc', sep = ''))
ctrl <- nc_open(paste(data_dir,'/forecasts/ensemble_fct/emos/ctrl_2006.nc', sep = ''))


prtb_data <- ncvar_get(prtb, "tp")
hres_data <- ncvar_get(hres, "tp")
ctrl_data <- ncvar_get(ctrl, "tp")


for (fold in 0:8){
  print(fold)
  year <- 11 + fold

  year_vals <- 365
  if (fold == 1 || fold == 5){
    year_vals <- 366
  }

  #train_tar_prev = nc_open(paste(data_dir , '/observation/emos/y_train_' , fold, ".nc", sep = ''))
  #obs_train_prev <- ncvar_get(train_tar_prev, "tp")
  #dim(obs_train_prev)[3]

  train_tar <- nc_open(paste(data_dir , '/observation/obs_precip_train.nc', sep = ''))
  obs_train_original <- ncvar_get(train_tar, "precipitationCal")


  train_start <- 2192
  train_end <- train_start + len_emos_train(fold) - 1 #dim(obs_train)[3]
  obs_train <- obs_train_original[, , train_start:train_end]

  if (fold == 8){
    val_tar <- nc_open(paste(data_dir , '/observation/obs_precip_test.nc', sep = ''))
    obs_test <- ncvar_get(val_tar, "precipitationCal")
    # 4383
  } else {
    test_start <- train_end + 1
    test_end <- test_start - 1 + year_vals
    obs_test <- obs_train_original[, , test_start:test_end]
  }

  start_emos <- 2
  end_emos <- dim(obs_train)[3] + 1

  #start_end_season <- ix_season(season, fold)
  #start_test <- start_end_season$start
  #end_test <- start_end_season$end


  start_emos2 <- end_emos + 1 #+ start_test
  end_emos2 <- end_emos + year_vals  # end_test #end_emos + year_vals

  prtb_data_train <- prtb_data[, , , start_emos:end_emos]
  prtb_data_test <- prtb_data[, , , start_emos2:end_emos2]
  ctrl_data_train <- ctrl_data[, , start_emos:end_emos]
  ctrl_data_test <- ctrl_data[, , start_emos2:end_emos2]
  hres_data_train <- hres_data[, , start_emos:end_emos]
  hres_data_test <- hres_data[, , start_emos2:end_emos2]

  #season_vals <- end_emos2 - start_emos2 + 1
  crps_vals <- matrix(, nrow = 813, ncol = year_vals)

  k <- 1
  for (i in 1:19){
    for (j in 1:61){
      if (!is.nan(lsm_bin[i, j])){
        print(k)
        obs_test_grid <- obs_test[j, i, ]
        obs_train_grid <- obs_train[j, i, ]

        prtb_data_train_grid <- prtb_data_train[, j, i, ]
        prtb_data_test_grid <- prtb_data_test[, j, i, ]

        ctrl_data_train_grid <- ctrl_data_train[j, i, ]
        ctrl_data_test_grid <- ctrl_data_test[j, i, ]

        hres_data_train_grid <- hres_data_train[j, i, ]
        hres_data_test_grid <-hres_data_test[j, i, ]

        crps_vals[k, ] <- emos_func(obs_train_grid, obs_test_grid, prtb_data_train_grid, prtb_data_test_grid, ctrl_data_train_grid, ctrl_data_test_grid, hres_data_train_grid, hres_data_test_grid)
        k <- k+1
      }
    }
  }

  write.table(crps_vals, paste(data_dir, '/results/prev_results_emos/emos_crps_', fold, '.txt', sep = ''), col.names = FALSE, row.names = FALSE)
}



season = 'JAS'
crps_folds <- rep(0, 9)

for (fold in 0:8){
  X <- read.table(paste(data_dir, '/results/prev_results_emos/emos_crps_', fold,'.txt', sep = ''))
  colnames(X) <- NULL
  X <- as.matrix(X)
  start_end_season <- ix_season(season, fold)
  crps_folds[(fold+1)] <- mean(X[, start_end_season$start:start_end_season$end])
}

crps_folds

write.table(crps_folds, paste(data_dir, '/results/emos_crps_',season ,'.txt', sep = ''), col.names = FALSE, row.names = FALSE)


