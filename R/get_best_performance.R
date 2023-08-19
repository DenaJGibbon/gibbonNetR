#' Extract Best Performance Results from Performance Tables
#'
#' Given the path to a directory of performance tables, this function reads in the tables,
#' combines them, and extracts the best performance results based on various criteria.
#'
#' @param performancetables.dir Path to the directory containing the performance tables.
#'
#' @return A list containing best F1 scores, best precision results, best recall results,
#'         and plots visualizing these metrics.
#' @importFrom purrr map_dfr
#' @importFrom readr read_csv
#' @importFrom ggpubr ggline ggscatter
#' @importFrom magrittr %>%
#' @export
get_best_performance <- function(performancetables.dir) {

  # Read all CSV files from the directory
  FrozenFiles <- list.files(performancetables.dir, full.names = TRUE)
  FrozenCombined <- suppressMessages(map_dfr(FrozenFiles, read_csv))

  unique_training_data <- unique(FrozenCombined$`Training Data`)

  best_f1_results <- data.frame()
  best_precision_results <- data.frame()
  best_recall_results <- data.frame()

  # Loop through each 'TrainingData' type and extract best performance metrics
  for (td in unique_training_data) {
    subset_data <- subset(FrozenCombined, `Training Data` == td)

    max_f1_row <- subset_data[which.max(subset_data$F1), ]
    best_f1_results <- rbind(best_f1_results, max_f1_row)

    max_precision_row <- subset_data[which.max(subset_data$Precision), ]
    best_precision_results <- rbind(best_precision_results, max_precision_row[which.max(max_precision_row$F1), ])

    max_recall_row <- subset_data[which.max(subset_data$Recall), ]
    best_recall_results <- rbind(best_recall_results, max_recall_row[which.max(max_recall_row$F1), ])
  }

  # Create visualizations
  f1_plot <- ggpubr::ggline(data = FrozenCombined, x = 'Threshold', y = 'F1', color = 'CNN Architecture', facet.by = 'N epochs')
  FrozenCombined$Recall <- round(FrozenCombined$Recall,1)
  pr_plot <- ggpubr::ggline(data = FrozenCombined, x = 'Recall', y = 'Precision', color = 'CNN Architecture', facet.by = 'N epochs')

  print('Best F1 results')
  print(best_f1_results)

  print('Best Precision results')
  print(best_precision_results)

  print('Best Recall results')
  print(best_recall_results)

  return(list(
    best_f1 = best_f1_results,
    best_precision = best_precision_results,
    best_recall = best_recall_results,
    f1_plot = f1_plot,
    pr_plot = pr_plot

  ))
}
