library(scoringRules)
library(ncdf4)
library(ensembleMOS)

test_date <- function(fold){
  year_fold <- 2011 + fold
  if (fold == 1 || fold == 5){
    dim_month <- 29
    dim_year <- 366
  } else {
    dim_month <- 28
    dim_year <- 365
  }
  month_length <- c(31, 31, dim_month, 31, 30, 31, 30, 31, 31, 30, 31, 30)
  all_vals <- c()
  for (val in month_length){
    all_vals <- c(all_vals, seq(val))
  }
  all_vals_str <- paste(all_vals)
  date_test <- rep(NA, dim_year)
  month <- rep(c( '12', '01', '02', '03','04', '05', '06', '07', '08', '09', '10', '11'),times=c(31, 31, dim_month, 31, 30, 31, 30, 31, 31, 30, 31, 30))

  for (k in 1:dim_year){
    day_val<- all_vals[k]
    if (day_val < 10){
      day <- paste('0',all_vals_str[k], sep='' )
    } else {
      day <- all_vals_str[k]
    }
    if (k < 31){
      date_test[k] <- paste(year_fold, month[k] ,  day, sep = '')
    } else {
      date_test[k] <- paste(year_fold+1, month[k] , day, sep = '')
    }
  }
  return(date_test)
}


emos_func <- function(y0, y1, p0, p1, c0, c1, h0, h1){


  ens_df <- as.data.frame(t(p0))
  ens_df$ctrl <- c0
  ens_df$hres <- h0
  ens_df$obs <- y0

  ens_df_test <- as.data.frame(t(p1))
  ens_df_test$ctrl <- c1
  ens_df_test$hres <- h1
  ens_df_test$obs <- y1

  date_train <- train_date(fold)
  date_test <- test_date(fold)

  Data_train <- ensembleData(
    forecasts = ens_df[,1:52],
    observations = ens_df$obs,
    forecastHour = 24,
    #forecastHour = hori * 24L + 6L,
    initializationTime = 00,
    dates = date_train,
    exchangeable = setNames(
      c(1, 2, rep(3, 50)),
      c("hres", "ctrl", paste0("V", 1:50))
    )
  )

  Data_test <- ensembleData(
    forecasts = ens_df_test[,1:52],
    observations = ens_df_test$obs,
    forecastHour = 24,
    #forecastHour = hori * 24L + 6L,
    initializationTime = 00,
    dates = date_test,
    exchangeable = setNames(
      c(1, 2, rep(3, 50)),
      c("hres", "ctrl", paste0("V", 1:50))
    )
  )

  fit <- fitMOSgev0(Data_train)
  crps_EMOS_ens = ensembleMOS::crps(fit=fit,ensembleData = Data_test)
  return(crps_EMOS_ens[, 2])
}

emos_paras <- function(y0, y1, p0, p1, c0, c1, h0, h1){


  ens_df <- as.data.frame(t(p0))
  ens_df$ctrl <- c0
  ens_df$hres <- h0
  ens_df$obs <- y0

  ens_df_test <- as.data.frame(t(p1))
  ens_df_test$ctrl <- c1
  ens_df_test$hres <- h1
  ens_df_test$obs <- y1

  date_train <- train_date(fold)
  date_test <- test_date(fold)

  Data_train <- ensembleData(
    forecasts = ens_df[,1:52],
    observations = ens_df$obs,
    forecastHour = 24,
    #forecastHour = hori * 24L + 6L,
    initializationTime = 00,
    dates = date_train,
    exchangeable = setNames(
      c(1, 2, rep(3, 50)),
      c("hres", "ctrl", paste0("V", 1:50))
    )
  )

  Data_test <- ensembleData(
    forecasts = ens_df_test[,1:52],
    observations = ens_df_test$obs,
    forecastHour = 24,
    #forecastHour = hori * 24L + 6L,
    initializationTime = 00,
    dates = date_test,
    exchangeable = setNames(
      c(1, 2, rep(3, 50)),
      c("hres", "ctrl", paste0("V", 1:50))
    )
  )

  fit <- fitMOSgev0(Data_train)

  paras_EMOS <- function(fit, Data_test){
    gini.md <- function(x,na.rm=FALSE)  {     ## Michael Scheuerer's code
      if(na.rm & any(is.na(x)))  x <- x[!is.na(x)]
      n <-length(x)
      return(4*sum((1:n)*sort(x,na.last=TRUE))/(n^2)-2*mean(x)*(n+1)/n)
    }
    S <- fit$s
    A <- fit$a
    B <- fit$B
    C <- fit$c
    D <- fit$d
    SHAPE <- fit$q
    MEAN <- SCALE <- LOC  <-  rep(NaN, dim(Data_test)[1])
    for (i in 1:dim(Data_test)[1]) {
      f = Data_test[i,]
      MEAN[i] <- as.numeric(c(A,B)%*%c(1,f)+S*mean(f==0, na.rm = TRUE))  #location of GEV
      SCALE[i] <- C + D*gini.md(f, na.rm = TRUE)  #scale of GEV
      LOC[i] <- as.numeric(MEAN[i] - SCALE[i]*(gamma(1-SHAPE)-1)/SHAPE)
    }
    return(list(LOC = LOC, SCALE = SCALE, SHAPE = rep(SHAPE, dim(Data_test)[1])))
  }

  Data_test_array <- as.matrix(Data_test[, 1:52])
  colnames(Data_test_array) <- NULL
  list_paras_emos <- paras_EMOS(fit, Data_test_array)
  # crps_EMOS_ens = ensembleMOS::crps(fit=fit,ensembleData = Data_test)
  return(list_paras_emos)
  #crps_EMOS_ens = ensembleMOS::crps(fit=fit,ensembleData = Data_test)
  #return(crps_EMOS_ens[, 2])
}

train_date <- function(fold){
  folds_year <- cumsum(c(0, 365, 366, 365, 365, 365, 366, 365, 365))
  dim_year <- 4*365 + 1 + folds_year[(fold + 1)]

  month_length <- c(31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30)
  all_vals <- c()
  for (val in month_length){
    all_vals <- c(all_vals, seq(val))
  }
  all_vals_str <- paste(all_vals)

  month <- rep(c( '12', '01', '02', '03','04', '05', '06', '07', '08', '09', '10', '11'),times=c(31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30))
  date_train <- rep(NA, dim_year)


  for (k in 1:365){
    day_val<- all_vals[k]
    if (day_val < 10){
      day <- paste('0',all_vals_str[k], sep='' )
    } else {
      day <- all_vals_str[k]
    }
    if (k < 31){
      date_train[(k+2*365+366)] <- paste('2009', month[k] , day, sep = '')
      date_train[(k+365+366)] <- paste('2008', month[k] , day, sep = '')
      #date_train[k+1*365] <- paste('2007', month[k] , day, sep = '')
      date_train[k] <- paste('2006', month[k] , day, sep = '')
    } else {
      date_train[(k+2*365+366)] <- paste('2010', month[k] , day, sep = '')
      date_train[(k+365+366)] <- paste('2009', month[k] , day, sep = '')
      #date_train[k+365] <- paste('2008', month[k] , day, sep = '')
      date_train[k] <- paste('2007', month[k] , day, sep = '')
    }
  }


  month2 <- rep(c( '12', '01', '02', '03','04', '05', '06', '07', '08', '09', '10', '11'),times=c(31, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30))

  month_length2 <- c(31, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30)
  all_vals2 <- c()
  for (val in month_length2){
    all_vals2 <- c(all_vals2, seq(val))
  }
  all_vals_str2 <- paste(all_vals2)

  for (k in 1:366){
    day_val<- all_vals2[k]
    if (day_val < 10){
      day <- paste('0',all_vals_str2[k], sep='' )
    } else {
      day <- all_vals_str2[k]
    }
    if (k < 31){
      date_train[(k+365)] <- paste('2007', month2[k] , day, sep = '')
      #print(day)
    } else {
      #print(day)
      date_train[(k+365)] <- paste('2008', month2[k] , day, sep = '')
    }
  }


  if (fold == 0){
    return(date_train)
  } else {
    for (k in 1:365){
      day_val<- all_vals[k]
      if (day_val < 10){
        day <- paste('0',all_vals_str[k], sep='' )
      } else {
        day <- all_vals_str[k]
      }
      if (k < 31){
        date_train[(k+3*365+366)] <- paste('2010', month[k] , day, sep = '')
      } else {
        date_train[(k+3*365+366)] <- paste('2011', month[k] , day, sep = '')
      }
    }
  }

  if (fold == 1){
    return(date_train)
  } else {
    for (k in 1:366){
      day_val<- all_vals2[k]
      if (day_val < 10){
        day <- paste('0',all_vals_str2[k], sep='' )
      } else {
        day <- all_vals_str2[k]
      }
      if (k < 31){
        date_train[(k+365)] <- paste('2011', month2[k] , day, sep = '')
        #print(day)
      } else {
        #print(day)
        date_train[(k+365)] <- paste('2012', month2[k] , day, sep = '')
      }
    }
  }

  if (fold == 2){
    return(date_train)
  } else {
    for (k in 1:365){
      day_val<- all_vals[k]
      if (day_val < 10){
        day <- paste('0',all_vals_str[k], sep='' )
      } else {
        day <- all_vals_str[k]
      }
      if (k < 31){
        date_train[(k+3*365+366)] <- paste('2012', month[k] , day, sep = '')
      } else {
        date_train[(k+3*365+366)] <- paste('2013', month[k] , day, sep = '')
      }
    }
  }


  if (fold == 3){
    return(date_train)
  } else {
    for (k in 1:365){
      day_val<- all_vals[k]
      if (day_val < 10){
        day <- paste('0',all_vals_str[k], sep='' )
      } else {
        day <- all_vals_str[k]
      }
      if (k < 31){
        date_train[(k+3*365+366)] <- paste('2013', month[k] , day, sep = '')
      } else {
        date_train[(k+3*365+366)] <- paste('2014', month[k] , day, sep = '')
      }
    }
  }
  if (fold == 4){
    return(date_train)
  } else {
    for (k in 1:365){
      day_val<- all_vals[k]
      if (day_val < 10){
        day <- paste('0',all_vals_str[k], sep='' )
      } else {
        day <- all_vals_str[k]
      }
      if (k < 31){
        date_train[(k+3*365+366)] <- paste('2014', month[k] , day, sep = '')
      } else {
        date_train[(k+3*365+366)] <- paste('2015', month[k] , day, sep = '')
      }
    }
  }

  if (fold == 5){
    return(date_train)
  } else {
    for (k in 1:366){
      day_val<- all_vals2[k]
      if (day_val < 10){
        day <- paste('0',all_vals_str2[k], sep='' )
      } else {
        day <- all_vals_str2[k]
      }
      if (k < 31){
        date_train[(k+365)] <- paste('2015', month2[k] , day, sep = '')
        #print(day)
      } else {
        #print(day)
        date_train[(k+365)] <- paste('2016', month2[k] , day, sep = '')
      }
    }
  }

  if (fold == 6){
    return(date_train)
  } else {
    for (k in 1:365){
      day_val<- all_vals[k]
      if (day_val < 10){
        day <- paste('0',all_vals_str[k], sep='' )
      } else {
        day <- all_vals_str[k]
      }
      if (k < 31){
        date_train[(k+3*365+366)] <- paste('2016', month[k] , day, sep = '')
      } else {
        date_train[(k+3*365+366)] <- paste('2017', month[k] , day, sep = '')
      }
    }
  }

  if (fold == 7){
    return(date_train)
  } else {
    for (k in 1:365){
      day_val<- all_vals[k]
      if (day_val < 10){
        day <- paste('0',all_vals_str[k], sep='' )
      } else {
        day <- all_vals_str[k]
      }
      if (k < 31){
        date_train[(k+3*365+366)] <- paste('2017', month[k] , day, sep = '')
      } else {
        date_train[(k+3*365+366)] <- paste('2018', month[k] , day, sep = '')
      }
    }
    return(date_train)
  }

}



ix_season <- function(season, fold){
  leap = 0
  if (fold == 1 || fold == 5){
    leap = 1
  }
  if (season == 'JAS'){
    start = 213 + leap
    end = start + 91
  } else if (season == 'MA'){
    start = 91 + leap
    end = start + 60
  } else if (season == 'MJ'){
    start = 152 + leap
    end = start + 60
  } else if (season == 'ON'){
    start = 305 + leap
    end = start + 60
  } else if (season == 'DJF'){
    start = 1
    end = start + leap + 89
  }else {
    stop("season no defined")
  }
  return(list(start = start, end =end))
}


len_emos_train <- function(fold){
  if (fold <= 1){
    leap = 0
  } else if (1 < fold && fold <= 5){
    leap = 1
  } else if (5 < fold && fold <= 8){
    leap = 2
  } else {
    stop('fold not defined')
  }
  start_len = 1461
  train_len = 1461 + leap + fold*365
  return(train_len)

}



emos_pit <- function(y0, y1, p0, p1, c0, c1, h0, h1){


  ens_df <- as.data.frame(t(p0))
  ens_df$ctrl <- c0
  ens_df$hres <- h0
  ens_df$obs <- y0

  ens_df_test <- as.data.frame(t(p1))
  ens_df_test$ctrl <- c1
  ens_df_test$hres <- h1
  ens_df_test$obs <- y1

  date_train <- train_date(fold)
  date_test <- test_date(fold)

  Data_train <- ensembleData(
    forecasts = ens_df[,1:52],
    observations = ens_df$obs,
    forecastHour = 24,
    #forecastHour = hori * 24L + 6L,
    initializationTime = 00,
    dates = date_train,
    exchangeable = setNames(
      c(1, 2, rep(3, 50)),
      c("hres", "ctrl", paste0("V", 1:50))
    )
  )

  Data_test <- ensembleData(
    forecasts = ens_df_test[,1:52],
    observations = ens_df_test$obs,
    forecastHour = 24,
    #forecastHour = hori * 24L + 6L,
    initializationTime = 00,
    dates = date_test,
    exchangeable = setNames(
      c(1, 2, rep(3, 50)),
      c("hres", "ctrl", paste0("V", 1:50))
    )
  )

  fit <- fitMOSgev0(Data_train)
  pit_EMOS = ensembleMOS::cdf(fit=fit,ensembleData = Data_test,values= ens_df_test$obs)
  vals <- diag(pit_EMOS)


  ix0 <- which(ens_df_test$obs <= 0.01)
  upits <- rep(0, length(ix0))
  for(ix in 1:length(ix0)){
    upits[ix] <- runif(1,0, vals[ix])
  }
  vals[ix0] <- upits

  return(vals)
}

