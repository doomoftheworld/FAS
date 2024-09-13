library(parallel)
library(doParallel)
library(foreach)
library(gtools)
library(boot)
library(RANN)
library(mvtnorm)
library(sensitivity)
library(pracma)

args <- commandArgs(trailingOnly = TRUE)

data_path <- args[1]
nb_class <- as.integer(args[2])
R_X_A <- read.csv(paste(data_path, "R_X_A.csv",sep =""))
R_X_B <- read.csv(paste(data_path, "R_X_B.csv",sep =""))
R_network_params_weight <- read.csv(paste(data_path, "R_network_params_weight.csv",sep =""))
R_network_params_bias <- read.csv(paste(data_path, "R_network_params_bias.csv",sep =""))

R_class_model <- function(X) {
  y <- as.matrix(X) %*% as.matrix(t(R_network_params_weight)) 
  y <- y + rep(as.matrix(R_network_params_bias), each = nrow(y))
  y
}

# Find the first and total order sobol indices
# Estimate the first order indices
result<-sobolMultOut(model=R_class_model, q=nb_class, R_X_A, R_X_B, 
                   MCmethod="soboljansen")
write.csv(result["S"], paste(data_path, "sobolmultout_fs.csv",sep =""), row.names=FALSE)
write.csv(result["T"], paste(data_path, "sobolmultout_tt.csv",sep =""), row.names=FALSE)
