# Function to compute BS decomposition

## Input:

# x: forecasts 
# y: observations

bs_unc <- function(obs){
  obs_pop <- mean(obs)
  return(mean((obs-obs_pop)^2))
}

CEP_pav <- function(x, y){
  ord <- order(x, -y)
  x <- x[ord]
  y <- y[ord]
  CEP_pav <- stats::isoreg(y)$yf
  bs_cal <- mean((CEP_pav - y)^2)
  bs_original <- mean((x - y)^2)
  mcb <- bs_original - bs_cal
  unc <- bs_unc(y)
  dsc <-  unc + mcb - bs_original
  return(data.frame('BS' = bs_original,'MSB' = mcb, 'DSC' = dsc, 'UNC' = unc))
}
