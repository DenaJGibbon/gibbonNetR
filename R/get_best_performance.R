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
get_best_performance <- function(performancetables.dir, model.type='multi',class='hornbill.helmeted',Thresh.val=0.5) {

  # Read all CSV files from the directory
  FrozenFiles <- list.files(performancetables.dir, full.names = TRUE)
  FrozenCombined <- suppressMessages(map_dfr(FrozenFiles, read_csv))

  if(model.type=='multi'){

    if( class %in% FrozenCombined$Class == FALSE){
      print(paste('Not detected', class, 'Here are the present classes:',unique(FrozenCombined$Class)))
      break
    }



    print(paste('Evaluating performance for', class, 'Here are the present classes:', paste(unique(FrozenCombined$Class))))

    FrozenCombined <-  droplevels(subset(FrozenCombined,Class==class))
  }

  unique_training_data <- unique(FrozenCombined$`Training Data`)

  best_f1_results <- data.frame()
  best_precision_results <- data.frame()
  best_recall_results <- data.frame()
  best_auc_results <- data.frame()

  FrozenCombined <- subset(FrozenCombined, Threshold >= Thresh.val)

  # Loop through each 'TrainingData' type and extract best performance metrics
  for (td in unique_training_data) {
    subset_data <- subset(FrozenCombined, `Training Data` == td)
    MaxF1 <- max(subset_data$F1,na.rm = TRUE)
    max_f1_row <- subset(subset_data, F1 == MaxF1)
    best_f1_results <- rbind(best_f1_results, max_f1_row)

    MaxPrecision <- max(subset_data$Precision,na.rm = TRUE)
    max_precision_row <- subset(subset_data, Precision == MaxPrecision)
    best_precision_results <- rbind(best_precision_results, max_precision_row[which.max(max_precision_row$F1), ])

    MaxRecall <-  max(subset_data$Recall,na.rm = TRUE)
    max_recall_row <- subset(subset_data, Recall == MaxRecall)
    best_recall_results <- rbind(best_recall_results, max_recall_row[which.max(max_recall_row$F1), ])

    max_auc_row <- subset_data[which.max(subset_data$AUC), ]
    best_auc_results <- rbind(best_auc_results, max_auc_row)
  }


  # Create visualizations
  f1_plot <- ggpubr::ggline(data = FrozenCombined, x = 'Threshold', y = 'F1',
                            color = 'CNN Architecture', facet.by = 'N epochs')+ ggtitle(paste('Results for', class, 'class'))
  FrozenCombined$Recall <- round(FrozenCombined$Recall,1)
  pr_plot <- ggpubr::ggline(data = FrozenCombined, x = 'Recall', y = 'Precision',
                            color = 'CNN Architecture', facet.by = 'N epochs')+ ggtitle(paste('Results for', class, 'class'))


  FrozenCombined$TPR <- FrozenCombined$Sensitivity
  FrozenCombined$FPR <- 1-FrozenCombined$Specificity
  FPRTPR_plot <-ggpubr::ggline(data = FrozenCombined,  x = 'FPR', y = 'TPR',
                               color = 'CNN Architecture', facet.by = 'N epochs',numeric.x.axis = TRUE)+
    geom_abline(slope=1,intercept=0,lty='dashed')+ coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))+ ggtitle(paste('Results for', class, 'class'))


  print('Best F1 results')
  print(best_f1_results$Class)
  print(as.data.frame(best_f1_results[,c(5:7,13:17)]))

  print('Best Precision results')
  print(as.data.frame(best_precision_results[,c(5:7,13:17)]))

  print('Best Recall results')
  print(as.data.frame(best_recall_results[,c(5:7,13:17)]))

  print('Best AUC results')
  print(as.data.frame(best_auc_results[c(5:7,13:15,17)]))

  return(list(
    best_f1 = best_f1_results,
    best_precision = best_precision_results,
    best_recall = best_recall_results,
    best_auc = best_auc_results,
    f1_plot = f1_plot,
    pr_plot = pr_plot,
    FPRTPR_plot =FPRTPR_plot
  ))
}
