# Installing the packages (uncomment if not installed)
# install.packages("party")
# install.packages("mlbench")
# install.packages("quantregForest")
# install.packages("Metrics")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("gridExtra")
# install.packages("randomForest")
# install.packages("ISLR")
# install.packages("caret")

# Load necessary packages
library(party)
library(mlbench)
library(quantregForest)
library(Metrics)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(randomForest)
library(ISLR)
library(caret)

cat("== Step 1: Processing the dataset and determining predictor variables == \n")

# Read and clean the NHANES dataset (this dataset is also attached in the github)
nhanes_clean <- read.csv("/Users/quintyokhuijsen/Downloads/cleaned_nhanes.csv")
# Remove variables that were similar to other variables
nhanes_clean2 <- select(nhanes_clean, -c(ID, SurveyYr, SexEver, BPDiaAve, BPSys1, BPDia1, BPSys2, BPDia2, BPSys3, BPDia3, Weight, BMI_WHO, DirectChol))
# Remove missing observations
nhanes <- na.omit(nhanes_clean2)

# Convert character columns to factor columns
character_cols <- sapply(nhanes, is.character)
nhanes[character_cols] <- lapply(nhanes[character_cols], as.factor)
cat("NHANES dataset has been cleaned and processed. \n")

# Define the target variable and predictors
target_var <- "BPSysAve"
selected_vars <- c("Age", "BMI", "PhysActive", "AlcoholYear", "TotChol")
nhanes_selected_vars <- nhanes %>% select(c(target_var, selected_vars))

# Function to compute conditional variable importance
compute_conditional_importance <- function(predictor, data, all_predictors) {
  # Create formula with the given predictor as the response variable
  formula <- as.formula(paste(predictor, "~", paste(setdiff(all_predictors, predictor), collapse = "+")))
  # Make the RF model
  rf_forest_cond <- randomForest(formula, data = data, importance = TRUE)
  # Compute the importance scores (the importance formula calculates the importance scores based on MSE)
  varimp_scores <- importance(rf_forest_cond, type = 1)
  varimp_scores <- varimp_scores[, 1]
  names(varimp_scores) <- rownames(importance(rf_forest_cond))
  return(varimp_scores)
}

# Function to extract cut points from the QRF Model
extract_cut_points <- function(qrf_model) {
  # List to store the cut points in for all trees
  cut_points <- list()
  
  for (tree_idx in 1:qrf_model$ntree) {
    # List to store the cut points in for current tree
    tree_cut_points <- list()
    # Get the number of nodes in the current tree
    num_nodes <- qrf_model$forest$ndbigtree[tree_idx]
    
    # Loop through each node in the current tree
    for (node_idx in 1:num_nodes) {
      # Make sure the node is a split node and not a leaf node
      if (qrf_model$forest$nodestatus[node_idx, tree_idx] != -1) {
        # Get the index of the variable used for splitting the node
        var_idx <- qrf_model$forest$bestvar[node_idx, tree_idx]
        # Get the split value at this node
        split_value <- qrf_model$forest$xbestsplit[node_idx, tree_idx]
        # Get the name of the variable that was used for this split
        var_name <- names(qrf_model$forest$ncat)[var_idx]
        
        # Check if the variable already has split values recorded
        if (!is.null(tree_cut_points[[var_name]])) {
          # If it already has split values recorded, the new split value should be added to that list
          tree_cut_points[[var_name]] <- c(tree_cut_points[[var_name]], split_value)
        } else {
          # Otherwise, a new list should be created to store the value of the current split
          tree_cut_points[[var_name]] <- split_value
        }
      }
    }
    
    # Loop through each variable that has a split in the current tree
    for (var_name in names(tree_cut_points)) {
      # Check if the variable already has split values recorded across all trees
      if (!is.null(cut_points[[var_name]])) {
        # Combine and keep only unique split values across all trees for this variable, otherwise it causes issues
        cut_points[[var_name]] <- unique(c(cut_points[[var_name]], tree_cut_points[[var_name]]))
      } else {
        # Otherwise, create the list of split values for this variable with the current tree's split values
        cut_points[[var_name]] <- tree_cut_points[[var_name]]
      }
    }
  }
  
  return(cut_points)
}

# Function to create a grid based on the data cut points
create_grid <- function(data, cut_points) {
  # Convert specified columns in data to numeric otherwise it causes issues
  numeric_data <- data %>% mutate(across(names(cut_points), as.numeric))
  # Create an empty dataframe with the same number of rows as data
  grid <- data.frame(matrix(ncol = 0, nrow = nrow(data)))
  
  # Loop through each variable name in cut_points from the above function
  for (var_name in names(cut_points)) {
    # Cut the numeric data into intervals using the cut points and store it in grid
    grid[[var_name]] <- cut(numeric_data[[var_name]], breaks = c(-Inf, sort(cut_points[[var_name]]), Inf), labels = FALSE)
  }
  
  return(grid)
}

# Quantile loss function
quantile_loss_function <- function(actual, predicted, alpha) {
  abs_diff <- abs(actual - predicted)
  weights <- ifelse(actual > predicted, alpha, 1 - alpha)
  mean(weights * abs_diff)
}

# Function to calculate conditional permutation importance
conditional_permutation_importance <- function(qrf_model, data, var_interest, n_permutations = 100, conditioning_vars, cut_points, threshold = 0.05, alpha) {
  # Function to compute out-of-bag accuracy
  oob_accuracy <- function(model, data, alpha) {
    predictions <- predict(model, data, what = alpha)
    accuracy <- quantile_loss_function(data[, target_var], predictions, alpha)
    return(accuracy)
  }
  
  # Compute baseline accuracy
  baseline_accuracy <- oob_accuracy(qrf_model, data, alpha)
  cat("Baseline accuracy at quantile", alpha, "is:", baseline_accuracy, "\n")
  
  # Get conditioning variables for the variable of interest
  conditional_vars <- conditioning_vars[[var_interest]]
  
  # If no conditioning variables, compute standard permutation importance
  if (length(conditional_vars) == 0) {
    permuted_accuracies <- numeric(n_permutations)
    for (i in 1:n_permutations) {
      permuted_data <- data
      permuted_data[, var_interest] <- sample(permuted_data[, var_interest])
      permuted_accuracy <- oob_accuracy(qrf_model, permuted_data, alpha)
      permuted_accuracies[i] <- permuted_accuracy
    }
    importance <- (mean(permuted_accuracies) - baseline_accuracy) / baseline_accuracy
    return(importance)
  }
  
  # Create grid for conditional permutation importance
  grid <- create_grid(data, cut_points)
  permuted_accuracies <- numeric(n_permutations)
  for (i in 1:n_permutations) {
    if (i %% 20 == 0) {
      cat("Permutation for CVI", i, "\n")
    }
    permuted_data <- data
    split_data <- split(permuted_data, grid[, conditional_vars])
    permuted_groups <- lapply(split_data, function(group) {
      if (nrow(group) > 1) {
        group[, var_interest] <- sample(group[, var_interest])
      }
      return(group)
    })
    permuted_data <- do.call(rbind, permuted_groups)
    permuted_accuracy <- oob_accuracy(qrf_model, permuted_data, alpha)
    permuted_accuracies[i] <- permuted_accuracy
  }
  importance <- (mean(permuted_accuracies) - baseline_accuracy) / baseline_accuracy
  cat("Conditional Variable Importance for --", var_interest, "-- at quantile", alpha, "is:", importance, "\n")
  return(importance)
}

# Function to calculate standard variable importance
usual_variable_importance <- function(qrf_model, data, var_interest, n_permutations = 100, alpha) {
  # Function to compute out-of-bag accuracy
  oob_accuracy <- function(model, data, alpha) {
    predictions <- predict(model, data, what = alpha)
    accuracy <- quantile_loss_function(data[, target_var], predictions, alpha)
    return(accuracy)
  }
  
  # Compute baseline accuracy
  baseline_accuracy <- oob_accuracy(qrf_model, data, alpha)
  
  # Compute standard permutation importance
  permuted_accuracies <- numeric(n_permutations)
  for (i in 1:n_permutations) {
    if (i %% 20 == 0) {
      cat("Permutation for VI", i, "\n")
    }
    permuted_data <- data
    permuted_data[, var_interest] <- sample(permuted_data[, var_interest])
    permuted_accuracy <- oob_accuracy(qrf_model, permuted_data, alpha)
    permuted_accuracies[i] <- permuted_accuracy
  }
  mean_decrease_accuracy <- (mean(permuted_accuracies) - baseline_accuracy) / baseline_accuracy
  cat("Standard Variable Importance for --", var_interest, "-- at quantile", alpha, "is:", mean_decrease_accuracy, "\n")
  return(mean_decrease_accuracy)
}

# Initialize result storage for cross-validation
quantiles <- c(0.005, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.995)
importance_results_cv <- data.frame(variable = character(), 
                                    conditional_importance = numeric(), 
                                    usual_importance = numeric(),
                                    quantile = numeric(),
                                    fold = numeric())

# Perform 5-fold cross-validation
folds <- createFolds(nhanes$BPSysAve, k = 5, list = TRUE)
for (fold in 1:5) {
  cat("++++ Processing fold ", fold, "+++++ \n")
  train_idx <- unlist(folds[-fold])
  test_idx <- unlist(folds[fold])
  
  nhanes_train <- nhanes_selected_vars[train_idx, ]
  nhanes_test <- nhanes_selected_vars[test_idx, ]
  
  # Determine the two most important conditioning variables for each predictor in the training set
  conditioning_vars <- list()
  for (predictor in selected_vars) {
    cat("Processing", predictor, "\n")
    varimp_scores <- compute_conditional_importance(predictor, nhanes_train, selected_vars)
    sorted_vars <- sort(varimp_scores, decreasing = TRUE)
    top_two_vars <- names(sorted_vars)[1:2]
    cat("Conditioning variables of ", predictor, "are ", top_two_vars, "\n")
    conditioning_vars[[predictor]] <- top_two_vars
  }
  
  for (q in quantiles) {
    cat("++++ Processing quantile ", q, "+++++ \n")
    qrf_model <- quantregForest(x = select(nhanes_train, -BPSysAve), y = nhanes_train$BPSysAve, ntree = 1000, mtry = 2, nodesize = 10)
    cut_points <- extract_cut_points(qrf_model)
    
    for (var in selected_vars) {
      print(paste("For :", var, "at quantile", q))
      conditional_importance <- conditional_permutation_importance(qrf_model, nhanes_test, var_interest = var, n_permutations = 100, conditioning_vars = conditioning_vars, cut_points = cut_points, threshold = 0.05, alpha = q)
      usual_importance <- usual_variable_importance(qrf_model, nhanes_test, var_interest = var, n_permutations = 100, alpha = q)
      importance_results_cv <- rbind(importance_results_cv, data.frame(variable = var, 
                                                                       conditional_importance = conditional_importance, 
                                                                       usual_importance = usual_importance,
                                                                       quantile = q,
                                                                       fold = fold))
    }
  }
}

# Aggregate results across folds
importance_results_aggregated <- importance_results_cv %>%
  group_by(variable, quantile) %>%
  summarize(conditional_importance = mean(conditional_importance),
            usual_importance = mean(usual_importance))

# Print aggregated results
print(importance_results_aggregated)
