#' Evaluate Model Performance on Image Data
#'
#' Given trained models and a set of images, this function evaluates the performance of the models.
#'
#' @param trained_models_dir Path to the directory containing trained models (.pt files).
#' @param image_data_dir Path to the directory containing image data for evaluation.
#' @param output_dir Path to the directory where the performance scores will be saved.
#'
#' @return Invisible NULL. The performance scores are written to the specified output directory.
#' @importFrom stringr str_split_fixed str_detect
#' @importFrom purrr %>%
#' @export
evaluate_trainedmodel_performance_multi <- function(trained_models_dir, image_data_dir, output_dir='data/',
                                                    noise.category='noise') {

  tryCatch({
  # List trained models
  trained_models <- list.files(trained_models_dir, pattern = '.pt', full.names = TRUE)

  # List image files
  image_files <- list.files(image_data_dir, recursive = TRUE, full.names = TRUE)
  image_files_short <- list.files(image_data_dir, recursive = TRUE, full.names = FALSE)

  # Loop through each trained model
  for (model_path in trained_models) {
    performance_scores <- data.frame()
    model <- luz_load(model_path)

    model_name <- basename(model_path)
    training_data <- str_split_fixed(model_name, pattern = '_', n = 4)[,2]
    n_epochs <- str_split_fixed(model_name, pattern = '_', n = 4)[,3]
    model_type <- str_split_fixed(str_split_fixed(model_name, pattern = '_', n = 4)[,4], pattern = '.pt', n = 2)[,1]

    print(paste('Training', model_type, 'N epochs=',n_epochs))
    # Evaluate model on each image file

    Folder <- sapply(image_files_short, function(x) dirname(x))

    # Define transforms based on model type
    if (str_detect(model_type, pattern = 'ResNet')) {
      transform_list <- . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_color_jitter() %>%
        transform_resize(256) %>%
        transform_center_crop(224) %>%
        transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    } else {
      transform_list <- . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_resize(size = c(224, 224)) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }

    test_ds <- image_folder_dataset(image_data_dir, transform = transform_list, target_transform = function(x) as.double(x) - 1)
    test_dl <- dataloader(test_ds, batch_size = 32, shuffle =FALSE)

    # Predict using
    Pred <- predict(model, test_dl)

    # Return the index of the max values (i.e. which class)
    PredMPS <- torch_argmax(Pred, dim = 2)

    # Save to cpu
    PredMPS <- as_array(torch_tensor(PredMPS, device = 'cpu'))

    # Convert to a factor
    modelPred<- as.factor(PredMPS)
    print(modelPred)

    # Calculate the probability associated with each class
    Probability <- as_array(torch_tensor(nnf_softmax(Pred, dim = 2), device = 'cpu'))

    # Find the index of the maximum value in each row
    max_prob_idx <- apply(Probability, 1, which.max)

    # Map the index to actual probability
    predicted_class_probability <- sapply(1:nrow(Probability), function(i) Probability[i, max_prob_idx[i]])

    # Convert the integer predictions to factor and then to character based on the levels
    modelNames <- factor(modelPred, levels = 1:length(class_names), labels = class_names)

    outputTable <- cbind.data.frame(modelNames, predicted_class_probability)
    colnames(outputTable) <- c('PredictedClass', 'Probability')
    outputTable$ActualClass <- Folder


    UniqueClasses <- unique(outputTable$ActualClass)
    Probability <- as.data.frame(Probability)
    colnames(Probability) <- UniqueClasses
    UniqueClasses <- UniqueClasses[-which(UniqueClasses == noise.category)]

    # Initialize data frames
    CombinedTempRow <- data.frame()
    TransferLearningCNNDF <- data.frame()
    thresholds <- seq(0.1, 1, 0.1)

    for (b in 1:length(UniqueClasses)) {

      outputTableSub <-outputTable
      outputTableSub$Probability <- Probability[,c(UniqueClasses[b] )]

      outputTableSub$ActualClass <-
        ifelse(outputTableSub$ActualClass==UniqueClasses[b],UniqueClasses[b],noise.category)

      for (threshold in thresholds) {
        PredictedClass <- ifelse((outputTableSub$Probability > threshold ), UniqueClasses[b], noise.category)

        Perf <- caret::confusionMatrix(
          as.factor(PredictedClass),
          as.factor(outputTableSub$ActualClass),
          mode = 'everything'
        )$byClass

        TempRow <- cbind.data.frame(
          t(Perf),
          trainingfolder,
          n_epochs,
          model_type
        )

        colnames(TempRow) <- c(
          "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
          "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
          "Detection Prevalence", "Balanced Accuracy",
          "Training Data",
          "N epochs",
          "CNN Architecture"
        )

        ROCRpred <- ROCR::prediction(predictions = outputTableSub$Probability, labels = outputTableSub$ActualClass)
        AUCval <- ROCR::performance(ROCRpred, 'auc')
        TempRow$AUC <- AUCval@y.values[[1]]
        TempRow$Threshold <- as.character(threshold)
        TempRow$Frozen <- unfreeze
        TempRow$Class <- UniqueClasses[b]
        TempRow$Class <- as.factor(TempRow$Class)
        CombinedTempRow <- rbind.data.frame(CombinedTempRow, TempRow)
      }
    }

    filename <- paste(output_dir,'performance_tables_multi_trained/', training_data, '_', n_epochs, '_', model_type, '_TransferLearningTrainedModel.csv', sep = '')

    filename_multi <- paste(output_dir, '/performance_tables_multi_trained_combined/', trainingfolder, '_', n_epochs, '_', model_type,'_TransferLearningCNNDFmulti.csv', sep = '')

    dir.create(paste(output_dir, '/performance_tables_multi_trained/', sep = ''))
    dir.create(paste(output_dir, '/performance_tables_multi_trained_combined/', sep = ''))

    write.csv(CombinedTempRow, filename, row.names = FALSE)

    Indexout <- which(outputTable$PredictedClass %in% outputTable$ActualClass==FALSE)
    outputTable$PredictedClass[Indexout] <- noise.category
    Perf <- caret::confusionMatrix(
      as.factor( droplevels(outputTable$PredictedClass)),
      as.factor(outputTable$ActualClass),
      mode = 'everything'
    )$byClass

    TempRow <- cbind.data.frame(
      (Perf),
      trainingfolder,
      n_epochs,
      model_type
    )

    colnames(TempRow) <- c(
      "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
      "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
      "Detection Prevalence", "Balanced Accuracy",
      "Training Data",
      "N epochs",
      "CNN Architecture"
    )

    TempRow$Class <- str_split_fixed(rownames(TempRow), pattern = ': ', n = 2)[, 2]

    write.csv(TempRow, filename_multi, row.names = FALSE)

    rm(model)
  }
  }, error = function(e) {
    # Handle the error here (e.g., print an error message)
    cat("Error occurred:", conditionMessage(e), "\n")
    # Optionally, you can log the error, save relevant information, or take other actions.
  })
}
