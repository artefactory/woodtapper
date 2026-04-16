
library(sirus)
library('profmem')
#library("ranger")

benchmark_model <- function(model_func, X_train, y_train, X_test) {

  if (!requireNamespace("profmem", quietly = TRUE)) {
    stop("Please install the 'profmem' package: install.packages('profmem')")
  }

  # --- Fit phase ---
  start_fit <- Sys.time()
  model <- model_func(X_train, y_train)
  fit_time <- as.numeric(difftime(Sys.time(), start_fit, units = "secs"))


  metrics <- list(
    fit_time = fit_time
  )

  list(metrics = metrics)
}



demo <- function(X_train,y_train,X_test) {

  # Define model function
  model_func <- function(X, y) {
    #glm(y ~ ., data = data.frame(X, y), family = binomial)
    sirus.m <- sirus.fit(data=X, y=y,type='classif', p0 = 0.0, q = 10, mtry = 14, num.trees = 1000, num.rule=25,#alpha=0.05
                         max.depth=2,num.threads = 5, replace = TRUE,sample.fraction=1.0, verbose = FALSE, seed = 0)
    return(sirus.m)
  }

  result <- benchmark_model(model_func, X_train, y_train, X_test)

  cat("=== Benchmark Results ===\n")
  cat("Fit time:", round(result$metrics$fit_time, 4), "s\n")
  return(result$metrics)
}
output_dir_data_set <- "paper/reproduce-exp/sim-data-time/" #simulation data for timing


X_train_original <-  read.csv(paste0(output_dir_data_set, "X_train.csv"), header = FALSE)
y_train_original <-  read.csv(paste0(output_dir_data_set, "y_train.csv"), header = FALSE)
X_test_original <-  read.csv(paste0(output_dir_data_set, "X_test.csv"), header = FALSE)
y_test_original <-  read.csv(paste0(output_dir_data_set, "y_test.csv"), header = FALSE)

y_train_original <- as.integer(unlist(y_train_original))
y_test_original <- as.integer(unlist(y_test_original))

list_time_samples <- c()
run_list <- c(0,1,2,3,4)
for (run in run_list) {
  cat("\n================ Starting run SAMPLES",run,"================\n")
  n_dim <- 200
  n_samples_train <- c(100000, 200000, 300000, 400000, 500000)
  curr_run_time <- c()
  for (n_samples_train in n_samples_train) {
    X_train <- X_train_original[1:n_samples_train, 1:n_dim]
    y_train <- y_train_original[1:n_samples_train]
    X_test <- X_test_original[, 1:n_dim]
    cat("\n--- Benchmarking with", n_samples_train, "training samples and dim=",n_dim," ---\n")
    res <- demo(X_train,y_train,X_test)
    curr_run_time <- c(curr_run_time, res$fit_time)
  }
  list_time_samples <- c(list_time_samples, curr_run_time)
}
# Save the list of times for samples
output_dir_Rules <- "paper/timing/times-csv/"
write.csv(list_time_samples, file = file.path(output_dir_Rules, "list_time_samples_r.csv"), row.names = FALSE, col.names = FALSE)

list_time_dims <- c()
run_list <- c(0,1,2,3,4)
for (run in run_list) {
  cat("\n================ Starting run DIMENSION",run,"================\n")
  n_samples_train <- 300000
  n_dims <- as.integer(c(15,25,50,75,100,125,150,175,200))
  curr_run_time <- c()
  for (n_dim in n_dims) {
    X_train <- X_train_original[1:n_samples_train, 1:n_dim]
    y_train <- y_train_original[1:n_samples_train]
    X_test <- X_test_original[, 1:n_dim]
    cat("\n--- Benchmarking with", n_samples_train, "training samples and dim=",n_dim," ---\n")
    res <- demo(X_train,y_train,X_test)
    curr_run_time  <- c(curr_run_time, res$fit_time)
  }
  list_time_dims <- c(list_time_dims, curr_run_time)
}
# Save the list of times for dimensions
write.csv(list_time_dims, file = file.path(output_dir_Rules, "list_time_dims_r.csv"), row.names = FALSE, col.names = FALSE)
