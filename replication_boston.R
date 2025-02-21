#This code is done on the Boston Housing dataset. If you wish to test it on another dataset you need to make a few modifications to the code.
# Installing packages
 # install.packages("quantregForest")
 # install.packages("mlbench")
 # install.packages("alr4")
 # install.packages("caret")
 # install.packages("Metrics")
 # install.packages("dplyr")
 # install.packages("ggplot2")
 # install.packages("quantreg")
 # install.packages("gridExtra")

# Loading necessary libraries
library(quantregForest)
library(quantreg)
library(mlbench)
library(alr4)
library(caret)
library(Metrics)
library(dplyr)
library(ggplot2)

#Set working directory to wherever your guide program is located, otherwise the tree-based models will not work
setwd("/Users/quintyokhuijsen/Desktop/guide")

# Loading the boston housing dataset from the mlbench package
data("BostonHousing")
boston <- BostonHousing
response_var_boston <- "medv"


#Permuting the dataset and cutting it into 5 equal parts (for random cross validation)
set.seed(123)
boston <- boston[sample(nrow(boston)), ]
folds_boston <- cut(seq(1, nrow(boston)), breaks=5, labels=FALSE)

#Store training and test data splits
boston_train <- list()
boston_test <- list()

#Creating the dataframes that store the loss results for each method
results_qrf_boston <- data.frame()
results_lqr_boston <- data.frame()
results_qqr_next_boston <- data.frame()
results_qqr_boston <- data.frame(Fold = integer(), Quantile = numeric(), Loss = numeric(), stringsAsFactors = FALSE)
results_trm_boston <- data.frame()
results_trc_boston <- data.frame()
results_trp_boston <- data.frame()

#Function for the quantile loss 
quantile_loss_function <- function(actual, predicted, alpha) {
  abs_diff <- abs(actual - predicted)
  weights <- ifelse(actual > predicted, alpha, 1 - alpha)
  mean(weights * abs_diff)
}

#The phrase that runs guide
guide_executable <- "./guide"

#Function for generating dsc files for TRM in guide
#These files should otherwise be manually created in order for guide to work
trm_generate_dsc_file <- function(fold_num, response_var) {
  
  #This creates a custom file name based on the fold (e.g. fold 1 would have the name boston_train_1.txt)
  train_file_name <- paste0("boston_train_", fold_num, ".txt")
  
  #This is what the structure of the dsc file should look like. It should start with the training data (since we use cross validation this is the training fold), then list all the variables with their type
  #If you wish to use another dataset, this is where you should change the variables for your own dataset
  dsc_content <- c(
    train_file_name,
    "NA",
    "1",
    paste0("1 crim n"),
    paste0("2 zn n"),
    paste0("3 indus n"),
    paste0("4 chas f"),
    paste0("5 nox n"),
    paste0("6 rm n"),
    paste0("7 age n"),
    paste0("8 dis n"),
    paste0("9 rad n"),
    paste0("10 tax n"),
    paste0("11 ptratio n"),
    paste0("12 b n"),
    paste0("13 lstat n"),
    paste0("14 ", response_var, " d")
  )
  
  # Then we end up with 5 dsc files for each of the 5 folds
  dsc_file_name <- paste0("boston_fold_", fold_num, ".dsc")
  writeLines(dsc_content, con = dsc_file_name)
}

#input file creation for TRM for guide 
#Normally when you run guide, the program automatically creates these input files for you based on your answers to several questions
#This method speeds up the process by generating all of the input files in one go, as doing it all manually takes a lot of time
trm_create_input_file <- function(fold, quantile) {
  # We format the quantile
  quantile_str <- sprintf("q%03d", quantile * 1000)
  
  # We create the file names based on the fold and quantile the program is running for
  output_file_name <- paste0("out_TRM_bos_", fold, "_", quantile_str, ".txt")
  dsc_file_name <- paste0("boston_", fold, ".dsc")
  latex_file_name <- paste0("TRM_bos_", fold, "_", quantile_str, ".tex")
  fitted_file_name <- paste0("TRM_bos_", fold, "_", quantile_str, "_fitted.txt")
  r_code_file_name <- paste0("predict_TRM_bos_", fold, "_", quantile_str, ".r")
  input_file_name <- paste0("in_TRM_bos_", fold, "_", quantile_str, ".txt")
  
  # Here the content of the input file is created. The structure of a general input file is basically copied and changed for each fold and quantile
  input_content <- c(
    "GUIDE       (do not edit this file unless you know what you are doing)",
    "  42.4      (version of GUIDE that generated this file)",
    " 1          (1=model fitting, 2=importance or DIF scoring, 3=data conversion)",
    paste0("\"", output_file_name, "\"  (name of output file)"),
    " 1          (1=one tree, 2=ensemble)",
    " 2          (1=classification, 2=regression, 3=propensity score tree)",
    " 2          (1=linear, 2=quantile, 3=Poisson, 4=censored response, 5=multiresponse or itemresponse, 6=longitudinal with T vars, 7=logistic)",
    " 1          (1=multiple linear, 2=best simple polynomial, 3=constant)",
    sprintf(" %.3f     (quantile)", quantile),
    " 1          (1=interaction tests, 2=skip them)",
    " 1          (0=tree with fixed no. of nodes, 1=prune by CV, 2=no pruning)",
    paste0("\"", dsc_file_name, "\"  (name of DSC file)"),
    "        10  (number of cross-validations)",
    " 1          (1=mean-based CV tree, 2=median-based CV tree)",
    "  0.250     (SE number for pruning)",
    " 1          (1=accept default splitting fraction, 2=change it)",
    " 1          (1=default max. number of split levels, 2=specify no. in next line)",
    " 1          (1=default min. node size, 2=specify min. value in next line)",
    " 2          (0=no LaTeX code, 1=tree without node numbers, 2=tree with node numbers)",
    paste0("\"", latex_file_name, "\" (latex file name)"),
    " 3          (0=all white,1=yellow-skyblue,2=yellow-purple,3=yellow-orange,4=orange-skyblue,5=yellow-red,6=orange-purple,7=grayscale)",
    " 1          (1=no storage, 2=store fit and split variables, 3=store split variables and values)",
    " 1          (1=do not save, 2=save regression coefs in a file)",
    " 2          (1=do not save fitted values and node IDs, 2=save in a file)",
    paste0("\"", fitted_file_name, "\" (file name for fitted values and node IDs)"),
    " 2          (1=do not write R function, 2=write R function)",
    paste0("\"", r_code_file_name, "\" (R code file)"),
    " 1          (rank of top variable to split root node)"
  )
  
  # We then save the input file for later use
  writeLines(input_content, con = input_file_name)
}

# Function to create TRC input files for GUIDE. This is done in the same way as we did for TRM above, but now with the answers specific to TRC.
trc_create_guide_input_files <- function(fold, quantile) {
  file_content <- paste(
    "GUIDE       (do not edit this file unless you know what you are doing)",
    "  42.4      (version of GUIDE that generated this file)",
    " 1          (1=model fitting, 2=importance or DIF scoring, 3=data conversion)",
    sprintf("\"out_TRC_bos_%d_q%03d.txt\"  (name of output file)", fold, quantile * 1000),
    " 1          (1=one tree, 2=ensemble)",
    " 2          (1=classification, 2=regression, 3=propensity score tree)",
    " 2          (1=linear, 2=quantile, 3=Poisson, 4=censored response, 5=multiresponse or itemresponse, 6=longitudinal with T vars, 7=logistic)",
    " 3          (1=multiple linear, 2=best simple polynomial, 3=constant)",
    " 1          (1=one quantile, 2=two quantiles)",
    sprintf(" 0.%03d     (quantile)", quantile * 1000),
    " 1          (1=interaction tests, 2=skip them)",
    " 1          (0=tree with fixed no. of nodes, 1=prune by CV, 2=no pruning)",
    sprintf("\"boston_%d.dsc\"  (name of DSC file)", fold),
    "        10  (number of cross-validations)",
    " 1          (1=mean-based CV tree, 2=median-based CV tree)",
    "  0.250     (SE number for pruning)",
    " 1          (1=accept default splitting fraction, 2=change it)",
    " 1          (1=default max. number of split levels, 2=specify no. in next line)",
    " 1          (1=default min. node size, 2=specify min. value in next line)",
    " 2          (0=no LaTeX code, 1=tree without node numbers, 2=tree with node numbers)",
    sprintf("\"TRC_bos_%d_q%03d.tex\" (latex file name)", fold, quantile * 1000),
    " 3          (0=all white,1=yellow-skyblue,2=yellow-purple,3=yellow-orange,4=orange-skyblue,5=yellow-red,6=orange-purple,7=grayscale)",
    " 1          (1=no storage, 2=store fit and split variables, 3=store split variables and values)",
    " 2          (1=do not save fitted values and node IDs, 2=save in a file)",
    sprintf("\"TRC_bos_%d_q%03d_fitted.txt\" (file name for fitted values and node IDs)", fold, quantile * 1000),
    " 2          (1=do not write R function, 2=write R function)",
    sprintf("\"predict_TRC_bos_%d_q%03d.r\" (R code file)", fold, quantile * 1000),
    " 1          (rank of top variable to split root node)",
    sep = "\n"
  )
  
  file_name <- sprintf("in_TRC_bos_%d_q%03d.txt", fold, quantile * 1000)
  writeLines(file_content, file_name)
  
}

# Function to create TRP input files for GUIDE This is done in the same way as we did for TRM and TRC above, but now with the answers specific to TRP.
trp_create_guide_input_files <- function(fold, quantile) {
  file_content <- paste(
    "GUIDE       (do not edit this file unless you know what you are doing)",
    "  42.4      (version of GUIDE that generated this file)",
    " 1          (1=model fitting, 2=importance or DIF scoring, 3=data conversion)",
    sprintf("\"out_TRP_bos_%d_q%03d.txt\"  (name of output file)", fold, quantile * 1000),
    " 1          (1=one tree, 2=ensemble)",
    " 2          (1=classification, 2=regression, 3=propensity score tree)",
    " 2          (1=linear, 2=quantile, 3=Poisson, 4=censored response, 5=multiresponse or itemresponse, 6=longitudinal with T vars, 7=logistic)",
    " 2          (1=multiple linear, 2=best simple polynomial, 3=constant)",
    " 1          (highest degree of polynomial model)",
    sprintf(" 0.%04d     (quantile)", quantile * 10000),
    " 1          (1=interaction tests, 2=skip them)",
    " 1          (0=tree with fixed no. of nodes, 1=prune by CV, 2=no pruning)",
    sprintf("\"boston_%d.dsc\"  (name of DSC file)", fold),
    "        10  (number of cross-validations)",
    " 1          (1=mean-based CV tree, 2=median-based CV tree)",
    "  0.250     (SE number for pruning)",
    " 1          (1=accept default splitting fraction, 2=change it)",
    " 1          (1=default max. number of split levels, 2=specify no. in next line)",
    " 1          (1=default min. node size, 2=specify min. value in next line)",
    " 2          (0=no LaTeX code, 1=tree without node numbers, 2=tree with node numbers)",
    sprintf("\"TRP_bos_%d_q%03d.tex\" (latex file name)", fold, quantile * 1000),
    " 3          (0=all white,1=yellow-skyblue,2=yellow-purple,3=yellow-orange,4=orange-skyblue,5=yellow-red,6=orange-purple,7=grayscale)",
    " 1          (1=no storage, 2=store fit and split variables, 3=store split variables and values)",
    " 2          (1=do not save, 2=save regressor names in a file)",
    sprintf("\"TRP_bos_%d_q%03d_reg.txt\" (regressor names file)", fold, quantile * 1000),
    " 2          (1=do not save fitted values and node IDs, 2=save in a file)",
    sprintf("\"TRP_bos_%d_q%03d_fitted.txt\" (file name for fitted values and node IDs)", fold, quantile * 1000),
    " 2          (1=do not write R function, 2=write R function)",
    sprintf("\"predict_TRP_bos_%d_q%03d.r\" (R code file)", fold, quantile * 1000),
    " 1          (rank of top variable to split root node)",
    sep = "\n"
  )
  
  file_name <- sprintf("in_TRP_bos_%d_q%03d.txt", fold, quantile * 1000)
  writeLines(file_content, file_name)
 
}

#Here I specify the quantiles used (these are the same as in the original QRF paper)
quantiles <- c(0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995)

#Starting the 5-fold cross validation loop
for (i in 1:5) {
  #Splitting the dataset for the 5 fold cross validation
  cat("\n")
  cat("+++++++++ Processing Fold:", i, "++++++++++++++++ \n")
  testIndexes_boston <- which(folds_boston == i, arr.ind = TRUE)
  boston_test[[i]] <- boston[testIndexes_boston, ]
  boston_train[[i]] <- boston[-testIndexes_boston, ]
  
  #Saving the training and test files for later use in guide. Remember the dsc files use the boston_train_ files, so it is necessary to save the folds
  write.table(boston_train[[i]], file = paste0("boston_train_", i, ".txt"), row.names = FALSE, col.names = FALSE, quote=FALSE)
  write.table(boston_test[[i]], file = paste0("boston_test_", i, ".txt"), row.names = FALSE, col.names = FALSE, quote=FALSE)
  
  #Preparing folds for training and testing
  train_fold_boston <- boston_train[[i]]
  test_fold_boston <- boston_test[[i]]
  
  x_train_boston <- train_fold_boston %>% select(-all_of(response_var_boston))
  x_test_boston <- test_fold_boston %>% select(-all_of(response_var_boston))
  
  y_train_boston <- train_fold_boston[[response_var_boston]]
  y_test_boston <- test_fold_boston[[response_var_boston]]
  
  #---QRF---
  #Here we build the QRF model and save it to the cq (conditional quantiles) dataframe
  qrf_boston <- quantregForest(x=x_train_boston, y=y_train_boston, ntree=1000, mtry=4, nodesize=10)
  
  cq_qrf_boston <- predict(qrf_boston, x_test_boston, what=quantiles)
  colnames(cq_qrf_boston) <- paste0("quantile= ", quantiles)
  
  
  #Evaluate quantile loss for each quantile of QRF
  for (q in quantiles) {
    quantile_name <- paste0("quantile= ", q)
   
    loss_qrf_boston <- quantile_loss_function(y_test_boston, cq_qrf_boston[, quantile_name], q)
    results_qrf_boston <- rbind(results_qrf_boston, data.frame(Quantile = q, Method = "QRF", Loss = loss_qrf_boston))
    
  }
  
  # --- LQR ------
  #Here we model the LQR method
  lqr_formula_boston <- as.formula(paste(response_var_boston, "~ ."))
  for (q in quantiles) {
    lqr_model_boston <- rq(lqr_formula_boston, data = train_fold_boston, tau = q)
    cq_lqr_boston <- predict.rq(lqr_model_boston, newdata = test_fold_boston)
    loss_lqr_boston <- quantile_loss_function(y_test_boston, cq_lqr_boston, q)
    results_lqr_boston <- rbind(results_lqr_boston, data.frame(Fold = i, Quantile = q, Method = "LQR", Loss = loss_lqr_boston))
  }
  
  #--- QQR ---
  #Here we model the QQR method
  cat("== QQR for fold ", i, " == \n")
  base_formula_boston <- lqr_formula_boston
  base_error_boston <- mean(results_lqr_boston$Loss)
  cat("Base model boston: ", deparse(base_formula_boston), "\n")
  cat("Base error boston: ", base_error_boston, "\n")
  
  predictors_boston <- setdiff(names(boston), response_var_boston)
  interaction_terms_boston <- combn(predictors_boston, 2, FUN = function(x) paste(x, collapse = ":"))
  
  #We keep track of the best qqr model and its error
  best_model_boston_qqr <- base_formula_boston
  best_cv_error_boston_qqr <- base_error_boston
  improved <- TRUE
  
  while (improved) {
    improved <- FALSE
    current_best_formula <- best_model_boston_qqr
    current_best_error <- best_cv_error_boston_qqr
    
    for (term in interaction_terms_boston) {
      # Here we add the different interaction terms to the model
      formula_with_interaction_boston <- as.formula(paste(deparse(current_best_formula), "+", term))
      
      # We fit all the QR models with the interaction term for each quantile
      for (q in quantiles) {
        qqr_next_boston <- rq(formula_with_interaction_boston, data = train_fold_boston, tau = q)
        cq_qqr_next_boston <- predict.rq(qqr_next_boston, test_fold_boston)
        loss_qqr_next_boston <- quantile_loss_function(y_test_boston, cq_qqr_next_boston, q)
        results_qqr_next_boston <- rbind(results_qqr_next_boston, data.frame(Fold = i, Quantile = q, Method = "QQR", Loss = loss_qqr_next_boston))
      }
      
      next_error_boston_qqr <- mean(results_qqr_next_boston$Loss[results_qqr_next_boston$Fold == i & results_qqr_next_boston$Quantile == q])
      
      # Update the best model if current one is better
      if (next_error_boston_qqr < current_best_error) {
        #cat("The added term is BETTER", "\n")
        current_best_error <- next_error_boston_qqr
        current_best_formula <- formula_with_interaction_boston
        improved <- TRUE
      } else {
        #These were just some debugging statements I added
        #cat("The added term is WORSE", "\n")
        #cat("With corresponding formula: ", deparse(formula_with_interaction_boston), "\n")
      }
    }

    
    if (improved) {
      #Again just a debuggin statement I added
      #cat("The current model is BETTER", "\n")
      best_cv_error_boston_qqr <- current_best_error
      best_model_boston_qqr <- current_best_formula
      #cat("This model is: ", deparse(best_model_boston_qqr), "and has error: ", best_cv_error_boston_qqr, "\n")
    } else {
      
      
    }
    
    for (q in quantiles) {
      quantile_specific_error <- mean(results_qqr_next_boston$Loss[results_qqr_next_boston$Fold == i & results_qqr_next_boston$Quantile == q])
      results_qqr_boston <- rbind(results_qqr_boston, data.frame(Fold = i, Quantile = q, Loss = quantile_specific_error))
    }
    
    cat("Final model: ", deparse(best_model_boston_qqr), "with error: ", best_cv_error_boston_qqr, "\n")
    cat("-------------------- \n")
  }
  
  #---- TRM ----
  #This is for the TRM method
  cat("== TRM for fold ", i, " == \n")
  
  #Here we generate the dsc files for each fold
  trm_generate_dsc_file(i, response_var_boston)
  cat("DSC file for fold ", i, " generated \n")
  
  #And we create the input files for each fold and quantile (this is why it is good to generate this process with Rstudio, otherwise you would have manually had to create 35 input files in guide)
  for (q in quantiles) {
    input_file_name <- trm_create_input_file(i, q)
    
    # Give the input file a name based on dataset, quantile and fold
    input_file_boston <- sprintf("in_TRM_bos_%d_q%03d.txt", i, q * 1000)
    
    #Run guide for all folds and quantiles
    #This was the command to use in the terminal to open guide
    guide_command <- paste(guide_executable, "<", input_file_boston)
    #Making sure that guide runs (this was a debuggin statement)
    print(paste("Running GUIDE with command:", guide_command))
    system(guide_command)
    
    # This is the R script file name corresponding to the dataset, fold and quantile
    r_script_boston <- sprintf("predict_TRM_bos_%d_q%03d.R", i, q * 1000)
    
    # Here we read the R script content
    r_script_content <- readLines(r_script_boston)
    
    # There were some issues with running the R script content, so I had to make some changes to the script
    #First of all the sep had to be changed
    r_script_content <- gsub("read.table\\((.*)\\)", "read.table(\\1, sep = \"\")", r_script_content)
    r_script_content <- gsub("newdata <- read.table\\(.*\\)", sprintf("newdata <- read.table('boston_test_%d.txt', header = FALSE, colClasses = 'character', sep = '')", i), r_script_content)
    #And the following had to be modified too
    r_script_content <- gsub("node <- NULL", "node <- NULL\npred <- NULL", r_script_content)

    
    # Here we had to convert the columns to numeric as this was causing issues
    r_script_content <- gsub("newdata <- transform\\(.*\\)", "", r_script_content)
    r_script_content <- gsub("newdata <- as.numeric\\(.*\\)", "", r_script_content)
    
    # the placement of "newdata" was also incorrect, so we changed that too
    #If you want to use another dataset, you need to change the variables here to your own variables too.
    r_script_content <- c(r_script_content, 
                          "# Convert columns to numeric",
                          "newdata <- transform(newdata,",
                          "                            crim = as.numeric(crim),",
                          "                            zn = as.numeric(zn),",
                          "                            indus = as.numeric(indus),",
                          "                            chas = as.numeric(chas),",
                          "                            nox = as.numeric(nox),",
                          "                            rm = as.numeric(rm),",
                          "                            age = as.numeric(age),",
                          "                            dis = as.numeric(dis),",
                          "                            rad = as.numeric(rad),",
                          "                            tax = as.numeric(tax),",
                          "                            ptratio = as.numeric(ptratio),",
                          "                            b = as.numeric(b),",
                          "                            lstat = as.numeric(lstat),",
                          "                            medv = as.numeric(medv))",
                          "# Saving predictions to a data frame ",
                          "predictions <- data.frame(node = node, pred = pred)",
                          "# And also writing them in a csv file to store them",
                          sprintf("write.csv(predictions, 'predictions_test_TRM_bos_%d_q%03d_TRM.csv', row.names = FALSE)", i, q * 1000),
                          "# We define again the quantile loss function",
                          "quantile_loss <- function(y, q, alpha) { ifelse(y > q, alpha * abs(y - q), (1 - alpha) * abs(y - q)) }",
                          "# Actual values",
                          "actual_values <- as.numeric(newdata$medv)",
                          "# We remove any NAs if necessary, as this sometimes causes issues",
                          "predicted_values <- predictions$pred",
                          "# The quantile level",
                          sprintf("alpha <- %.3f", q),
                          "# Calculate the quantile loss for each prediction",
                          "loss_values <- quantile_loss(actual_values, predicted_values, alpha)",
                          "# Calculate the mean quantile loss",
                          "mean_loss <- mean(loss_values, na.rm = TRUE)",
                          "# Print the mean quantile loss",
                          "print(paste('Mean Quantile Loss:', mean_loss))",
                          sprintf("write(mean_loss, 'loss_fold_bos_%d_q%03d_TRM.txt')", i, q * 1000))
    
    # Write the changed content back to the original R script
    writeLines(r_script_content, r_script_boston)
    
    # Run this new and changed r script
    r_command <- paste("Rscript", r_script_boston)
    #checking if command is correct
    print(paste("Running R script with command:", r_command))
    system(r_command)
    
    #Here we can see the loss calculated by the r script
    loss_file_boston <- sprintf("loss_fold_%d_q%03d_TRM.txt", i, q * 1000)
    cat("Loss for fold ", i, "and quantile " , q, " : ", loss_file_boston, "\n")
    
    loss_value <- as.numeric(readLines(loss_file_boston))
    results_trm_boston <- rbind(results_trm_boston, data.frame(Fold = i, Quantile = q, Loss = loss_value))
    
  }
  #---TRC---
  #Generate input files for TRC (repeat what you did for TRM but now for TRC)
  for (q in quantiles) {
    input_file_name <- trc_create_guide_input_files(i, q)
    input_file_boston <- sprintf("in_TRC_bos_%d_q%03d.txt", i, q * 1000)
    
    guide_command <- paste(guide_executable, "<", input_file_boston)
    print(paste("Running GUIDE with command:", guide_command))
    system(guide_command)
    
    r_script <- sprintf("predict_TRC_bos_%d_q%03d.r", i, q * 1000)
    
    # Again, we had to make changes to this R script as it was not working in the first place
    r_script_content <- readLines(r_script)
    r_script_content <- gsub("read.table\\((.*)\\)", "read.table(\\1, sep = \"\")", r_script_content)
    r_script_content <- gsub("newdata <- read.table\\(.*\\)", sprintf("newdata <- read.table('boston_test_%d.txt', header = FALSE, colClasses = 'character', sep = '')", i), r_script_content)
    
    #If you want to use another dataset, you need to change the variables here to your own variables too.
    r_script_content <- c(
      r_script_content,
      "# Converting columns to numeric",
      "newdata <- transform(newdata,",
      "                      crim = as.numeric(crim),",
      "                      zn = as.numeric(zn),",
      "                      indus = as.numeric(indus),",
      "                      chas = as.numeric(chas),",
      "                      nox = as.numeric(nox),",
      "                      rm = as.numeric(rm),",
      "                      age = as.numeric(age),",
      "                      dis = as.numeric(dis),",
      "                      rad = as.numeric(rad),",
      "                      tax = as.numeric(tax),",
      "                      ptratio = as.numeric(ptratio),",
      "                      b = as.numeric(b),",
      "                      lstat = as.numeric(lstat),",
      "                      medv = as.numeric(medv))",
      "# Saving the predictions to a data frame ",
      "predictions <- data.frame(node = node, pred = pred)",
      "# Write predictions to a file to store them just in case",
      sprintf("write.csv(predictions, 'predictions_test_%d_q%03d_trc_bos.csv', row.names = FALSE)", i, q * 1000),
      "# The quantile loss function",
      "quantile_loss <- function(y, q, alpha) { ifelse(y > q, alpha * abs(y - q), (1 - alpha) * abs(y - q)) }",
      "# Actual values",
      "actual_values <- as.numeric(newdata$medv)",
      "# Remove missing values",
      "predicted_values <- predictions$pred",
      "# The quantile level:",
      sprintf("alpha <- %.3f", q),
      "# Calculate the quantile loss for each prediction",
      "loss_values <- quantile_loss(actual_values, predicted_values, alpha)",
      "# Calculate the mean quantile loss",
      "mean_loss <- mean(loss_values, na.rm = TRUE)",
      "# Print the mean quantile loss",
      "print(paste('Mean Quantile Loss:', mean_loss))",
      sprintf("write(mean_loss, 'loss_fold_%d_q%03d_trc_bos.txt')", i, q * 1000)
    )
    
    writeLines(r_script_content, r_script)
    
    # Here again we run the r script
    r_command <- paste("Rscript", r_script)
    print(paste("Running R script with command:", r_command))
    system(r_command)
    
    #and this r script calculates the average loss per fold and quantile
    loss_file_boston <- sprintf("loss_fold_%d_q%03d_TRC_bos.txt", i, q * 1000)
    cat("Loss for fold ", i, "and quantile " , q, " : ", loss_file_boston, "\n")
    
    loss_value <- as.numeric(readLines(loss_file_boston))
    results_trc_boston <- rbind(results_trc_boston, data.frame(Fold = i, Quantile = q, Loss = loss_value))
  }
  
  #--- TRP---
  # Generate input files for TRP (same thing as for TRM and TRC)
  for (q in quantiles) {
    input_file_name <- trp_create_guide_input_files(i, q)
    input_file_boston <- sprintf("in_TRP_bos_%d_q%03d.txt", i, q * 1000)
    
    guide_command <- paste(guide_executable, "<", input_file_boston)
    print(paste("Running GUIDE with command:", guide_command))
    system(guide_command)
    
    
    r_script <- sprintf("predict_TRP_bos_%d_q%03d.r", i, q * 1000)
    
    # We again make modifcations in the r script like in TRM and TRC
    r_script_content <- readLines(r_script)
    r_script_content <- gsub("read.table\\((.*)\\)", "read.table(\\1, sep = \"\")", r_script_content)
    r_script_content <- gsub("boston_train_[0-9]+.txt", sprintf("boston_test_%d.txt", i), r_script_content)
    
    #If you want to use another dataset, you need to change the variables here to your own variables too.
    r_script_content <- c(
      r_script_content,
      "# Converting columns to numeric",
      "newdata <- transform(newdata,",
      "                      crim = as.numeric(crim),",
      "                      zn = as.numeric(zn),",
      "                      indus = as.numeric(indus),",
      "                      chas = as.numeric(chas),",
      "                      nox = as.numeric(nox),",
      "                      rm = as.numeric(rm),",
      "                      age = as.numeric(age),",
      "                      dis = as.numeric(dis),",
      "                      rad = as.numeric(rad),",
      "                      tax = as.numeric(tax),",
      "                      ptratio = as.numeric(ptratio),",
      "                      b = as.numeric(b),",
      "                      lstat = as.numeric(lstat),",
      "                      medv = as.numeric(medv))",
      "# Saving the predictions to a data frame",
      "predictions <- data.frame(node = node, fitvar = fitvar, pred = pred)",
      "# Write predictions to a file to store them",
      sprintf("write.csv(predictions, 'predictions_test_%d_q%03d_TRP_bos.csv', row.names = FALSE)", i, q * 1000),
      "# The quantile loss function",
      "quantile_loss <- function(y, q, alpha) { ifelse(y > q, alpha * abs(y - q), (1 - alpha) * abs(y - q)) }",
      "# Actual values",
      "actual_values <- as.numeric(newdata$medv)",
      "# Handle missing values",
      "predicted_values <- predictions$pred",
      "# Specify the quantile level",
      sprintf("alpha <- %.3f", q),
      "# Calculate the quantile loss for each prediction",
      "loss_values <- quantile_loss(actual_values, predicted_values, alpha)",
      "# Calculate the mean quantile loss",
      "mean_loss <- mean(loss_values, na.rm = TRUE)",
      "# Print the mean quantile loss",
      "print(paste('Mean Quantile Loss:', mean_loss))",
      sprintf("write(mean_loss, 'loss_fold_%d_q%03d_TRP_bos.txt')", i, q * 1000)
    )
    
    writeLines(r_script_content, r_script)
    
    # Run the modified prediction script
    r_command <- paste("Rscript", r_script)
    #checking if the r command is correct
    print(paste("Running R script with command:", r_command))
    system(r_command)
    
    #and again here we can see the loss value that the r script calculates 
    loss_file_boston <- sprintf("loss_fold_%d_q%03d_TRP_bos.txt", i, q * 1000)
    loss_value <- as.numeric(readLines(loss_file_boston))
    results_trp_boston <- rbind(results_trp_boston, data.frame(Fold = i, Quantile = q, Loss = loss_value))
  }
  
  
}

#--- QRF ---
#Summarising QRF mean loss results
summary_qrf_boston <- results_qrf_boston %>%
  group_by(Quantile, Method) %>%
  summarise(MeanLoss = mean(Loss, na.rm = TRUE, scientific = FALSE), .groups = 'drop')
print(summary_qrf_boston)

#--- LQR ---
#Summarising LQR mean loss results
summary_lqr_boston <- results_lqr_boston %>%
  group_by(Quantile, Method) %>%
  summarise(MeanLoss = mean(Loss, na.rm = TRUE, scientific = FALSE), .groups = 'drop')
print(summary_lqr_boston)

#--- QQR ---
#Summarising QQR mean loss results
print("Results QQR:")
print(results_qqr_next_boston)
summary_qqr_boston <- results_qqr_next_boston %>%
  group_by(Quantile, Method) %>%
  summarise(MeanLoss = mean(Loss, na.rm = TRUE), .groups = 'drop')
print(summary_qqr_boston)

#--- TRM ---
#Summarising TRM mean loss results
#The R script for the 0.005 quantile was not returning the loss function, so we manually put it in the dataframe
results_trm_boston[1, "Loss"] <- 0.0536067997911971
summary_trm_boston <- results_trm_boston %>%
  group_by(Quantile) %>%
  summarise(MeanLoss = mean(Loss, na.rm = TRUE), .groups = 'drop')
print(summary_trm_boston)

#--- TRC ---
#Summarising TRC mean loss results
summary_trc_boston <- results_trc_boston %>%
  group_by(Quantile) %>%
  summarise(MeanLoss = mean(Loss, na.rm = TRUE), .groups = 'drop')
print(summary_trc_boston)

#--- TRP ---
#Summarising TRP mean loss results
summary_trp_boston <- results_trp_boston %>%
  group_by(Quantile) %>%
  summarise(MeanLoss = mean(Loss, na.rm = TRUE), .groups = 'drop')
print(summary_trp_boston)

# Combine all summary dataframes with an additional column for the method
summary_qrf_boston <- summary_qrf_boston %>% select(Quantile, MeanLoss) %>% rename(QRF = MeanLoss)
summary_lqr_boston <- summary_lqr_boston %>% select(Quantile, MeanLoss) %>% rename(LQR = MeanLoss)
summary_qqr_boston <- summary_qqr_boston %>% select(Quantile, MeanLoss) %>% rename(QQR = MeanLoss)
summary_trm_boston <- summary_trm_boston %>% select(Quantile, MeanLoss) %>% rename(TRM = MeanLoss)
summary_trc_boston <- summary_trc_boston %>% select(Quantile, MeanLoss) %>% rename(TRC = MeanLoss)
summary_trp_boston <- summary_trp_boston %>% select(Quantile, MeanLoss) %>% rename(TRP = MeanLoss)

# Join all summaries by quantile
final_summary_boston <- summary_qrf_boston %>%
  left_join(summary_lqr_boston, by = "Quantile") %>%
  left_join(summary_qqr_boston, by = "Quantile") %>%
  left_join(summary_trm_boston, by = "Quantile") %>%
  left_join(summary_trc_boston, by = "Quantile") %>%
  left_join(summary_trp_boston, by = "Quantile")

print(final_summary_boston)

# Function to calculate bootstrap confidence intervals
bootstrap_ci <- function(data, n_bootstrap = 1000) {
  quantiles <- unique(data$Quantile)
  results <- list()
  
  for (q in quantiles) {
    quantile_data <- data %>% filter(Quantile == q) %>% select(Loss)
    
    bootstrap_means <- numeric(n_bootstrap)
    for (i in 1:n_bootstrap) {
      bootstrap_sample <- sample(quantile_data$Loss, size = nrow(quantile_data), replace = TRUE)
      bootstrap_means[i] <- mean(bootstrap_sample)
    }
    
    ci_lower <- quantile(bootstrap_means, 0.025)
    ci_upper <- quantile(bootstrap_means, 0.975)
    
    results[[as.character(q)]] <- list(mean_loss = mean(quantile_data$Loss), ci_lower = ci_lower, ci_upper = ci_upper)
  }
  
  return(results)
}

#To make it reproducible, set the seed
set.seed(123)

# Calculating bootstrap intervals for all methods (except QRF)
bootstrap_lqr_boston <- bootstrap_ci(results_lqr_boston)
boostrap_qqr_boston <- bootstrap_ci(results_qqr_boston)
boostrap_trc_boston <- bootstrap_ci(results_trc_boston)
boostrap_trm_boston <- bootstrap_ci(results_trm_boston)
boostrap_trp_boston <- bootstrap_ci(results_trp_boston)

#Create a dataframe that stores the mean loss and bootstrap intervals of all methods and quantiles
final_results_loss_bootstrap_boston <- data.frame(Method = character(),
                                                  Quantile = numeric(),
                                                  Average_Loss = numeric(),
                                                  Upper_Interval = numeric(),
                                                  Lower_Interval = numeric(),
                                                  stringsAsFactors = FALSE)
# Adding QRF average loss to dataframe (no bootstrap intervals)
for (i in 1:nrow(summary_qrf_boston)) {
  final_results_loss_bootstrap_boston <- rbind(final_results_loss_bootstrap_boston, data.frame(
    Method = "QRF",
    Quantile = summary_qrf_boston$Quantile[i],
    Average_Loss = summary_qrf_boston$QRF[i],
    Upper_Interval = NA,  # No upper interval for QRF
    Lower_Interval = NA,  # No lower interval for QRF
    stringsAsFactors = FALSE
  ))
}

# Adding LQR average loss and confidence intervals to dataframe
for (quantile in names(bootstrap_lqr_boston)) {
  quantile_result <- bootstrap_lqr_boston[[quantile]]
  final_results_loss_bootstrap_boston <- rbind(final_results_loss_bootstrap_boston, data.frame(
    Method = "LQR",
    Quantile = as.numeric(quantile),
    Average_Loss = quantile_result$mean_loss,
    Upper_Interval = quantile_result$ci_upper,
    Lower_Interval = quantile_result$ci_lower,
    stringsAsFactors = FALSE
  ))
}

# Adding QQR average loss and confidence intervals to dataframe
for (quantile in names(boostrap_qqr_boston)) {
  quantile_result <- boostrap_qqr_boston[[quantile]]
  final_results_loss_bootstrap_boston <- rbind(final_results_loss_bootstrap_boston, data.frame(
    Method = "QQR",
    Quantile = as.numeric(quantile),
    Average_Loss = quantile_result$mean_loss,
    Upper_Interval = quantile_result$ci_upper,
    Lower_Interval = quantile_result$ci_lower,
    stringsAsFactors = FALSE
  ))
}

# Adding TRC average loss and confidence intervals to dataframe
for (quantile in names(boostrap_trc_boston)) {
  quantile_result <- boostrap_trc_boston[[quantile]]
  final_results_loss_bootstrap_boston <- rbind(final_results_loss_bootstrap_boston, data.frame(
    Method = "TRC",
    Quantile = as.numeric(quantile),
    Average_Loss = quantile_result$mean_loss,
    Upper_Interval = quantile_result$ci_upper,
    Lower_Interval = quantile_result$ci_lower,
    stringsAsFactors = FALSE
  ))
}

# Adding TRM average loss and confidence intervals to dataframe
for (quantile in names(boostrap_trm_boston)) {
  quantile_result <- boostrap_trm_boston[[quantile]]
  final_results_loss_bootstrap_boston <- rbind(final_results_loss_bootstrap_boston, data.frame(
    Method = "TRM",
    Quantile = as.numeric(quantile),
    Average_Loss = quantile_result$mean_loss,
    Upper_Interval = quantile_result$ci_upper,
    Lower_Interval = quantile_result$ci_lower,
    stringsAsFactors = FALSE
  ))
}

# Adding TRP average loss and confidence intervals to dataframe
for (quantile in names(boostrap_trp_boston)) {
  quantile_result <- boostrap_trp_boston[[quantile]]
  final_results_loss_bootstrap_boston <- rbind(final_results_loss_bootstrap_boston, data.frame(
    Method = "TRP",
    Quantile = as.numeric(quantile),
    Average_Loss = quantile_result$mean_loss,
    Upper_Interval = quantile_result$ci_upper,
    Lower_Interval = quantile_result$ci_lower,
    stringsAsFactors = FALSE
  ))
}

#To print the final results, including the average loss and bootstrap confidence intervals of all quantiles and methods
print(final_results_loss_bootstrap_boston)

library(ggplot2)
library(gridExtra)


# Each method group gets a different colour in the graph
method_colors <- c("QRF" = "black", "LQR" = "green", "QQR" = "green",
                   "TRC" = "red", "TRM" = "red", "TRP" = "red")

# From left to right, the graph should show QRF, LQR, QQR, TRC, TRM and TRP
final_results_loss_bootstrap_boston$Method <- factor(final_results_loss_bootstrap_boston$Method, 
                                                     levels = c("QRF", "LQR", "QQR", "TRC", "TRM", "TRP"))

# Function to create the graph that shows the average loss, with the bootstrap intervals as vertical confidence bound lines
plot_function <- function(data, alpha) {
  qrf_avg_loss <- data %>% filter(Method == "QRF") %>% select(Average_Loss) %>% distinct() %>% pull()
  
  ggplot(data, aes(x = Method, y = Average_Loss, color = Method)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = Lower_Interval, ymax = Upper_Interval), width = 0.2) +
    
    #QRF should have a black horizonal line through it
    geom_hline(yintercept = qrf_avg_loss, linetype = "dashed", color = "black") +
    scale_color_manual(values = method_colors) +
    ggtitle(paste("α =", alpha)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title.y = element_blank(),  # Remove vertical axis title
          axis.title.x = element_blank(),  # Remove horizontal axis title
          legend.position = "none")
}



# Create the graphs for each quantile and method
plots <- lapply(unique(final_results_loss_bootstrap_boston$Quantile), function(quantile) {
  plot_function(final_results_loss_bootstrap_boston %>% filter(Quantile == quantile), quantile)
})

# Arrange the graphs in a grid with specified order
final_graph_boston <- grid.arrange(grobs = plots, ncol = length(unique(final_results_loss_bootstrap_boston$Quantile)))

# Saving the final graph
ggsave("loss_bootstrap_boston_graph.png", final_graph_boston)