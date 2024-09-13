library(parallel)
library(doParallel)
library(foreach)
library(gtools)
library(boot)
library(RANN)
library(mvtnorm)
library(sensitivity)

args <- commandArgs(trailingOnly = TRUE)

data_path <- args[1]
nb_class <- as.integer(args[2])
R_X_A <- read.csv(paste(data_path, "R_X_A.csv",sep =""))
R_X_B <- read.csv(paste(data_path, "R_X_B.csv",sep =""))
R_network_params <- read.csv(paste(data_path, "R_network_params.csv",sep =""))

R_nb_neurons <- ncol(R_X_A)
R_network_params[R_network_params$classId == '1',paste("weight_","0", sep="")]

R_current_class_id <- "0"
R_class_model <- function(X) {
  y <- 0
  for (j in 0:(R_nb_neurons-1)) {
    y <- y + R_network_params[R_network_params$classId == R_current_class_id,paste("weight_",as.character(j), sep="")]*X[,j+1]
  }
  y <- y + R_network_params[R_network_params$classId == R_current_class_id,"bias"]
  y
}

# Find the first and total order sobol indices
for (class_id in 0:(nb_class-1)) {
  # Modify the current class
  R_current_class_id <- as.character(class_id)
  # Estimate the first order indices
  x_fs<-sobolEff(model=R_class_model, X1=R_X_A, X2=R_X_B, nboot=0, order=1)
  write.csv(x_fs["S"], paste(data_path, "sobolEff_fs_", R_current_class_id, ".csv",sep =""), row.names=FALSE)
  rm(x_fs)
  # Estimate the total order indices
  x_tt<-sobolEff(model=R_class_model, X1=R_X_A, X2=R_X_B, nboot=0, order=0)
  write.csv(x_tt["S"], paste(data_path, "sobolEff_tt_", R_current_class_id, ".csv",sep =""), row.names=FALSE)
  rm(x_tt)
}