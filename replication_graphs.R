#install.packages("ggplot2")
#install.packages("gridExtra")
#install.packages("dplyr")

library(ggplot2)
library(gridExtra)
library(dplyr)

# +===+ Graph for average loss of methods with bootstrap confidence intervals for Boston Housing dataset +===+
#You can adapt this to another dataset

# Each method should have the same colour as in the original QRF paper
method_colors <- c("QRF" = "black", "LQR" = "green", "QQR" = "green",
                   "TRC" = "red", "TRM" = "red", "TRP" = "red")

boston_results <- transform(final_results_loss_bootstrap_boston, Dataset = "Boston Housing")

# These methods should also have the same order as in the original QRF dataset
boston_results$Method <- factor(boston_results$Method, levels = c("QRF", "LQR", "QQR", "TRC", "TRM", "TRP"))

# Function for plotting the graph of the average loss, with a black horizontal line through QRF
plot_function <- function(data, alpha) {
  qrf_avg_loss <- data %>% filter(Method == "QRF") %>% select(Average_Loss) %>% distinct() %>% pull()
  
  ggplot(data, aes(x = Method, y = Average_Loss, color = Method)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = Lower_Interval, ymax = Upper_Interval), width = 0.2) +
    geom_hline(yintercept = qrf_avg_loss, linetype = "dashed", color = "black") +
    scale_color_manual(values = method_colors) +
    ggtitle(paste("α =", alpha)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 16),  
          strip.text.x = element_text(size = 18, face = "bold"),  
          strip.text.y = element_text(size = 18, face = "bold"),  
          strip.placement = "outside",  
          axis.title.y = element_blank(),  
          axis.title.x = element_blank(),  
          legend.position = "none")
}

# Create all the graphs with the average loss per quantile and method and bootstrap confidence bounds 
combined_plot <- ggplot(boston_results, aes(x = Method, y = Average_Loss, color = Method)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = Lower_Interval, ymax = Upper_Interval), width = 0.2) +
  geom_hline(data = boston_results %>% filter(Method == "QRF"), 
             aes(yintercept = Average_Loss), linetype = "dashed", color = "black") +
  scale_color_manual(values = method_colors) +
  facet_grid(~ Quantile, scales = "free_y", switch = "y") +  
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 16),  
        strip.text.x = element_text(size = 18, face = "bold"),  
        strip.text.y = element_text(size = 18, face = "bold"),  
        strip.placement = "outside", 
        axis.title.y = element_blank(),  
        axis.title.x = element_blank(),  
        legend.position = "none")

# Save the graph
ggsave("boston_loss_bootstrap_combined_graph_y.png", combined_plot, width = 20, height = 20)

# +===+ Graph for average loss of methods with bootstrap confidence intervals WITH added noise for Boston Housing dataset +===+
# You can adapt this to another dataset if need be

boston_results_noise <- transform(final_results_loss_bootstrap_boston_noise, Dataset = "Boston Housing")

# Put the methods in the correct order from left to right
boston_results_noise$Method <- factor(boston_results_noise$Method, levels = c("QRF", "LQR", "QQR", "TRC", "TRM", "TRP"))

# Function for plotting the graph of the average loss, with a black horizontal line through QRF
plot_function <- function(data, alpha) {
  qrf_avg_loss <- data %>% filter(Method == "QRF") %>% select(Average_Loss) %>% distinct() %>% pull()
  
  ggplot(data, aes(x = Method, y = Average_Loss, color = Method)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = Lower_Interval, ymax = Upper_Interval), width = 0.2) +
    geom_hline(yintercept = qrf_avg_loss, linetype = "dashed", color = "black") +
    scale_color_manual(values = method_colors) +
    ggtitle(paste("α =", alpha)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 16),  
          strip.text.x = element_text(size = 18, face = "bold"),  
          strip.text.y = element_text(size = 18, face = "bold"),  
          strip.placement = "outside",  
          axis.title.y = element_blank(),  
          axis.title.x = element_blank(),  
          legend.position = "none")
}

# Create all the graphs with the average loss per quantile and method and bootstrap confidence bounds with noise for Boston Housing
combined_plot_noise <- ggplot(boston_results_noise, aes(x = Method, y = Average_Loss, color = Method)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = Lower_Interval, ymax = Upper_Interval), width = 0.2) +
  geom_hline(data = boston_results_noise %>% filter(Method == "QRF"), 
             aes(yintercept = Average_Loss), linetype = "dashed", color = "black") +
  scale_color_manual(values = method_colors) +
  facet_grid(~ Quantile, scales = "free_y", switch = "y") +  
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 16),  
        strip.text.x = element_text(size = 18, face = "bold"),  
        strip.text.y = element_text(size = 18, face = "bold"),  
        strip.placement = "outside", 
        axis.title.y = element_blank(),  
        axis.title.x = element_blank(),  
        legend.position = "none")
# Save the graph
ggsave("boston_noise_loss_bootstrap_combined_graph_y.png", combined_plot_noise, width = 20, height = 20)

# +===+ Graph for the prediction intervals for Boston Housing dataset +===+

# Load the Boston Housing dataset
data("BostonHousing")
boston <- BostonHousing
response_var_boston <- "medv"

# Perform 5-fold cross-validation
#Since we use the same seed the folds will be the same as in the replication_boston code
set.seed(123)
folds <- cut(seq(1, nrow(boston)), breaks = 5, labels = FALSE)

predictions_list <- list()
for (i in 1:5) {
  test_indexes <- which(folds == i, arr.ind = TRUE)
  test_data <- boston[test_indexes, ]
  train_data <- boston[-test_indexes, ]
  
  qrf_model <- quantregForest(x = train_data[, -14], y = train_data$medv, ntree = 500)
  predictions <- predict(qrf_model, newdata = test_data[, -14], what = c(0.025, 0.5, 0.975))
  
  pred_df <- data.frame(
    observed = test_data$medv,
    fitted = predictions[, "quantile= 0.5"],
    lower = predictions[, "quantile= 0.025"],
    upper = predictions[, "quantile= 0.975"]
  )
  predictions_list[[i]] <- pred_df
}

# Combine predictions from all folds
combined_predictions <- do.call(rbind, predictions_list)

# Calculate prediction interval length and sort for the right plot
combined_predictions <- combined_predictions %>%
  mutate(interval_length = upper - lower) %>%
  arrange(interval_length)

# subtracting the mean of the prediction interval
combined_predictions <- combined_predictions %>%
  mutate(observed_centered = observed - (upper + lower) / 2,
         lower_centered = lower - (upper + lower) / 2,
         upper_centered = upper - (upper + lower) / 2)

# Add an ordered index for the right plot
combined_predictions <- combined_predictions %>%
  mutate(ordered_index = 1:n())

#The plot for the prediction intervals
left_plot <- ggplot(combined_predictions, aes(x = fitted, y = observed)) +
  geom_errorbar(aes(ymin = lower, ymax = upper), color = "grey", alpha = 0.5, width = 0.3) +
  geom_segment(aes(xend = fitted, y = lower, yend = lower), color = "black") +
  geom_segment(aes(xend = fitted, y = upper, yend = upper), color = "black") +
  geom_point(color = "red") +
  labs(x = "fitted values (conditional median)", y = "observed values") +
  theme_minimal()

#The plot for the ordered prediction intervals
right_plot <- ggplot(combined_predictions, aes(x = ordered_index, y = observed_centered)) +
  geom_ribbon(aes(ymin = lower_centered, ymax = upper_centered), fill = "grey", alpha = 0.5) +
  geom_segment(aes(xend = ordered_index, y = lower_centered, yend = lower_centered), color = "black") +
  geom_segment(aes(xend = ordered_index, y = upper_centered, yend = upper_centered), color = "black") +
  geom_point(color = "red") +
  labs(x = "ordered samples", y = "observed values and prediction intervals (centered)") +
  theme_minimal()

# Combine the two plots
combined_plot <- grid.arrange(left_plot, right_plot, ncol = 2)

# Save the plot
ggsave("Boston_Housing_Prediction_Intervals.png", combined_plot, width = 14, height = 6)

