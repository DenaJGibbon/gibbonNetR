#' Extract Best Performance Results from Performance Tables
#'
#' Given the path to a directory of performance tables, this function reads in the tables,
#' combines them, and extracts the best performance results based on various criteria.
#'
#' @param performancetables.dir Path to the directory containing the performance tables.
#' @param model.type Type of model architecture. If 'multi' then will treat as multiclass, otherwise will treat as binary.
#' @param class Specific class for evaluation.
#' @param Thresh.val Threshold value for evaluation.
#'
#' @return A list containing best F1 scores, best precision results, best recall results,
#'         and plots visualizing these metrics.
#' @importFrom purrr map_dfr
#' @importFrom readr read_csv
#' @importFrom ggpubr ggline ggscatter
#' @importFrom magrittr %>%
#'

#' @examples
#' {
#' # Simulate data for performance tables
#' set.seed(123)
#'
#' #' Set directory
#' performance_tables_dir <- paste(tempdir(),"/example_performance_tables/", sep='')
#'
#' #' Create directory for performance tables (NOTE THIS IS FOR TESTING ONLY)
#' dir.create(performance_tables_dir, showWarnings = FALSE, recursive = TRUE)
#'
#' #' Define list of model architectures
#' architectures <- c("alexnet", "vgg16", "vgg19")
#'
#' #' Define list of training datasets
#' training_datasets <- c("Dataset1", "Dataset2", "Dataset3")
#'
#' #' Create performance tables
#' for (arch in architectures) {
#'   for (td in training_datasets) {
#'     #' Generate random performance metrics
#'     metrics <- data.frame(
#'       Class = rep(c("hornbill.helmeted", "other.class"), each = 5),
#'       "Training Data" = rep(td, 10),
#'       "CNN Architecture" = rep(arch, 10),
#'       Threshold = runif(10, 0, 1),
#'       F1 = runif(10, 0, 1),
#'       Precision = runif(10, 0, 1),
#'       Recall = runif(10, 0, 1),
#'       AUC = runif(10, 0, 1),
#'       `N epochs` = rep(c(10, 20, 30), each = 10)
#'     )
#'
#'    # Reassign column names
#'     colnames(metrics) <- c("Class",
#'     "Training Data", "CNN Architecture",
#'     "Threshold", "F1", "Precision",
#'     "Recall", "AUC", "N epochs")
#'     #' Write data to CSV file
#'     filename <- paste0(performance_tables_dir, arch, "_", td, ".csv")
#'     write.csv(metrics, filename, row.names = FALSE)
#'   }
#' }
#'
#'
#' #' Call the function with default parameters
#' results <- get_best_performance(performancetables.dir = performance_tables_dir, )
#'
#'  # NOTE: Results will not make sense as it is random
#' #' message the best F1 scores
#' message("Best F1 scores:")
#' print(results$best_f1)
#'
#' #' message the best precision results
#' message("Best precision results:")
#' print(results$best_precision)
#'
#' #' message the best recall results
#' message("Best recall results:")
#' print(results$best_recall)
#'
#' #' message the best AUC results
#' message("Best AUC results:")
#' print(results$best_auc)
#'
#' #' Plot F1 scores
#' print(results$f1_plot)
#'
#' #' Plot precision-recall curve
#' print(results$pr_plot)
#' }
#' @importFrom utils write.csv read.csv
#' @export
#'
get_best_performance <- function(performancetables.dir,
                                 model.type = 'multi',
                                 class = 'hornbill.helmeted',
                                 Thresh.val = 0.5) {
  # Read all CSV files from the directory
  FrozenFiles <- list.files(performancetables.dir, full.names = TRUE)
  FrozenCombined <- suppressMessages(map_dfr(FrozenFiles, read_csv))

  if (model.type == 'multi') {
    if (!class %in% FrozenCombined[["Class"]]) {
      message(paste(
        'Not detected',
        class,
        'Here are the present classes:',
        unique(FrozenCombined[["Class"]])
      ))
      return(NULL)  # Use return() instead of break
    }

    message(paste(
      'Evaluating performance for',
      class,
      'Here are the present classes:',
      paste(unique(FrozenCombined[["Class"]]))
    ))

    FrozenCombined <- droplevels(FrozenCombined[FrozenCombined[["Class"]] == class, ])
  }

  unique_training_data <- unique(FrozenCombined[["Training Data"]])

  best_f1_results <- data.frame()
  best_precision_results <- data.frame()
  best_recall_results <- data.frame()
  best_auc_results <- data.frame()

  FrozenCombined <- FrozenCombined[FrozenCombined[["Threshold"]] >= Thresh.val, ]

  # Loop through each 'Training Data' type and extract best performance metrics
  for (td in unique_training_data) {
    subset_data <- FrozenCombined[FrozenCombined[["Training Data"]] == td, ]

    max_f1_row <- subset_data[which.max(subset_data[["F1"]]), ]
    best_f1_results <- rbind(best_f1_results, max_f1_row)

    max_precision_row <- subset_data[which.max(subset_data[["Precision"]]), ]
    best_precision_results <- rbind(best_precision_results, max_precision_row)

    max_recall_row <- subset_data[which.max(subset_data[["Recall"]]), ]
    best_recall_results <- rbind(best_recall_results, max_recall_row)

    max_auc_row <- subset_data[which.max(subset_data[["AUC"]]), ]
    best_auc_results <- rbind(best_auc_results, max_auc_row)
  }

  FrozenCombined[["Threshold"]] <- as.numeric(round(FrozenCombined[["Threshold"]], 1))

  # Create visualizations
  f1_plot <- ggpubr::ggline(
    data = FrozenCombined,
    x = "Threshold",
    y = "F1",
    color = "CNN Architecture",
    facet.by = "N epochs"
  ) + ggtitle(paste("Results for", class, "class"))

  FrozenCombined[["Recall"]] <- round(FrozenCombined[["Recall"]], 1)
  pr_plot <- ggpubr::ggline(
    data = FrozenCombined,
    x = "Recall",
    y = "Precision",
    color = "CNN Architecture",
    facet.by = "N epochs"
  ) + ggtitle(paste("Results for", class, "class"))

  return(list(
    best_f1 = best_f1_results,
    best_precision = best_precision_results,
    best_recall = best_recall_results,
    best_auc = best_auc_results,
    f1_plot = f1_plot,
    pr_plot = pr_plot
  ))
}
