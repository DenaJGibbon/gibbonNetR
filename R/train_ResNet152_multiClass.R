#' Train a Multi-Class ResNet152 Model
#'
#' This function trains an ResNet152 model on a given dataset for multi-class image classification.
#' The trained model, along with performance metrics and other metadata, are saved to disk.
#'
#' @param input.data.path Character. The path to the input training data folder. Sub-folders should correspond to class labels.
#' @param test.data Character. The path to the input test data folder. Sub-folders should correspond to class labels.
#' @param unfreeze Logical. If TRUE, all layers of the pretrained ResNet152 will be unfrozen for retraining. Default is TRUE.
#' @param epoch.iterations Integer. The number of epochs to train the model for. Default is 1.
#' @param early.stop Character. If "yes", early stopping will be applied during training. Default is "yes".
#' @param output.base.path Character. The base path where output files will be saved. Default is 'data/'.
#' @param trainingfolder Character. A descriptor for naming the output folder.
#' @param learning_rate Numeric. Specify the desired learning rate.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{Output_Path}: The path where the model and metrics are saved.
#'   \item \code{Test_Eval}: Evaluation metrics on the test data.
#' }
#'
#' @seealso \code{\link[torch]{nn_module}}, \code{\link[torch]{dataloader}}, \code{\link[torch]{nn_bce_with_logits_loss}}
#' @examples
#' \dontrun{
#'   train_ResNet152_multiClass(
#'     input.data.path = "path/to/training/data",
#'     test.data = "path/to/test/data",
#'     unfreeze = TRUE,
#'     epoch.iterations = 10,
#'     early.stop = "yes",
#'     output.base.path = "data/output/",
#'     trainingfolder = "experiment_1"
#'   )
#' }
#' @export
#' @importFrom stringr str_replace
#' @importFrom readr write_csv
#' @importFrom tibble tibble
#' @importFrom magrittr %>%
#' @importFrom caret confusionMatrix
#' @importFrom ROCR prediction performance
#' @importFrom ggpubr ggline
#' @importFrom torch cuda_is_available nn_module nn_bce_with_logits_loss
#' @importFrom luz setup fit
#' @importFrom torchvision transform_to_tensor transform_resize transform_normalize transform_color_jitter
#'

train_ResNet152_multiClass <- function(
    input.data.path,
    test.data,
    unfreeze = TRUE,
    epoch.iterations = 1,
    early.stop = 'yes',
    output.base.path = 'data/',
    trainingfolder,
    learning_rate=0.01,
    noise.category = 'noise'
) {
  device <- if (cuda_is_available()) "cuda" else "cpu"

  to_device <- function(x, device) {
    x$to(device = device)
  }

  metadata <- tibble(
    Model_Name = "ResNet152",
    Training_Data_Path = input.data.path,
    Test_Data_Path = test.data,
    Output_Path = output.base.path,
    Device_Used = device,
    EarlyStop = early.stop,
    Layers_Ununfrozen = unfreeze,
    Epochs = epoch.iterations
  )

  write_csv(metadata, paste0(output.base.path, "ResNet152model_metadata.csv"))

  # Data loaders setup
  transform_func <- . %>%
    torchvision::transform_to_tensor() %>%
    transform_resize(256) %>%
    transform_center_crop(224) %>%
    torchvision::transform_color_jitter() %>%
    torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))

  train_ds <- image_folder_dataset(file.path(input.data.path, 'train'), transform = transform_func)
  valid_ds <- image_folder_dataset(file.path(input.data.path, "valid"), transform = transform_func)

  train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
  valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)

  # Read class names
  class_names <- sort(unique(attr(train_ds$class_to_idx, "names")))
  num_classes <- length(class_names)
  cat('Detected classes:', paste(class_names, collapse = ', '), '\n')

  for (a in 1:length(epoch.iterations)) {
    cat('Training ResNet152\n')
    n.epoch <- epoch.iterations[a]

    # Define the model
    net <- nn_module(
      initialize = function() {
        self$model <- model_resnet152(pretrained = TRUE)
        for (par in self$parameters) {
          par$requires_grad_(unfreeze)
        }
        self$model$fc <- nn_sequential(
          nn_linear(self$model$fc$in_features, 1024),
          nn_relu(),
          nn_linear(1024, 1024),
          nn_relu(),
          nn_linear(1024, num_classes)
        )
      },
      forward = function(x) {
        self$model(x)
      }
    )

    # Define the optimizer, loss function, and metrics
    fitted <- net %>%
      luz::setup(
        loss = nn_cross_entropy_loss(),
        optimizer = optim_adam,
        metrics = list(
          luz_metric_accuracy(),
          luz_metric_multiclass_auroc()
        )
      )

    # Training the model
    if (early.stop == 'yes') {
      modelResNet152Gibbon <- fitted %>%
        fit(train_dl, epochs = n.epoch, valid_data = valid_dl,
            callbacks = list(
              luz_callback_early_stopping(patience = 2),
              luz_callback_lr_scheduler(
                lr_one_cycle,
                max_lr =learning_rate,
                epochs = n.epoch,
                steps_per_epoch = length(train_dl),
                call_on = "on_batch_end"
              ),
              luz_callback_csv_logger(paste(output.base.path, trainingfolder, n.epoch, "logs_ResNet152.csv", sep = '_'))
            ),
            verbose = TRUE
        )
    } else {
      modelResNet152Gibbon <- fitted %>%
        fit(train_dl, epochs = n.epoch, valid_data = valid_dl,
            callbacks = list(
              luz_callback_lr_scheduler(
                lr_one_cycle,
                max_lr =learning_rate,
                epochs = n.epoch,
                steps_per_epoch = length(train_dl),
                call_on = "on_batch_end"
              ),
              luz_callback_csv_logger(paste(output.base.path, trainingfolder, n.epoch, "logs_ResNet152.csv", sep = '_'))
            ),
            verbose = TRUE
        )
    }

    luz_save(modelResNet152Gibbon, paste(output.base.path, trainingfolder, n.epoch, "modelResNet152.pt", sep = '_'))

    TempCSV.ResNet152 <- read.csv(paste(output.base.path, trainingfolder, n.epoch, "logs_ResNet152.csv", sep = '_'))
    ResNet152.loss <- TempCSV.ResNet152[nrow(TempCSV.ResNet152), ]$loss

    LossPlot <- ggline(data = TempCSV.ResNet152, x = 'epoch', y = 'loss', color = 'set')
    AUCPlot <- ggline(data = TempCSV.ResNet152, x = 'epoch', y = 'auc', color = 'set')

    print(cowplot::plot_grid(LossPlot,AUCPlot))

    # Calculate performance metrics for ResNet152 -------------------------------------
    dir.create(paste(output.base.path, 'performance_tables', sep = ''))
    dir.create(paste(output.base.path, 'performance_tables_multi', sep = ''))

    test_ds <- image_folder_dataset(
      file.path(test.data, "test"),
      transform = transform_func
    )

    test_dl <- dataloader(test_ds, batch_size = 32, shuffle = FALSE, drop_last = FALSE)

    # Get the list of image files
    imageFiles <- list.files(paste(test.data, '/', 'test', sep = ''), recursive = TRUE, full.names = TRUE)
    imageFileShort <- list.files(paste(test.data, '/', 'test', sep = ''), recursive = TRUE, full.names = FALSE)
    Folder <- str_split_fixed(imageFileShort, pattern = '/', n = 2)[, 1]
    imageFileShort <- str_split_fixed(imageFileShort, pattern = '/', n = 2)[, 2]

    # Prepare output tables
    outputTableResNet152 <- data.frame()

    # Predict using ResNet152
    ResNet152Pred <- predict(modelResNet152Gibbon, test_dl)

    # Return the index of the max values (i.e. which class)
    PredMPS <- torch_argmax(ResNet152Pred, dim = 2)

    # Save to cpu
    PredMPS <- as_array(torch_tensor(PredMPS, device = 'cpu'))

    # Convert to a factor
    modelResnetGibbonPred <- as.factor(PredMPS)
    print(modelResnetGibbonPred)

    # Calculate the probability associated with each class
    Probability <- as_array(torch_tensor(nnf_softmax(ResNet152Pred, dim = 2), device = 'cpu'))

    # Find the index of the maximum value in each row
    max_prob_idx <- apply(Probability, 1, which.max)

    # Map the index to actual probability
    predicted_class_probability <- sapply(1:nrow(Probability), function(i) Probability[i, max_prob_idx[i]])

    # Convert the integer predictions to factor and then to character based on the levels
    modelResnetGibbonNames <- factor(modelResnetGibbonPred, levels = 1:length(class_names), labels = class_names)

    outputTableResNet152 <- cbind.data.frame(modelResnetGibbonNames, predicted_class_probability)
    colnames(outputTableResNet152) <- c('PredictedClass', 'Probability')
    outputTableResNet152$ActualClass <- Folder

    # Save the output table as CSV file
    write.csv(outputTableResNet152, paste(output.base.path, trainingfolder, n.epoch, "output_ResNet152.csv", sep = '_'), row.names = FALSE)

    UniqueClasses <- unique(outputTableResNet152$ActualClass)
    Probability <- as.data.frame(Probability)
    colnames(Probability) <- UniqueClasses
    UniqueClasses <- UniqueClasses[-which(UniqueClasses == noise.category)]

    # Initialize data frames
    CombinedTempRow <- data.frame()
    TransferLearningCNNDF <- data.frame()
    thresholds <- seq(0.1, 1, 0.1)

    for (b in 1:length(UniqueClasses)) {


      outputTableResNet152Sub <-outputTableResNet152
      outputTableResNet152Sub$Probability <- Probability[,c(UniqueClasses[b] )]

      outputTableResNet152Sub$ActualClass <-
        ifelse(outputTableResNet152Sub$ActualClass==UniqueClasses[b],UniqueClasses[b],noise.category)

      for (threshold in thresholds) {
        ResNet152PredictedClass <- ifelse((outputTableResNet152Sub$Probability > threshold ), UniqueClasses[b], noise.category)

        ResNet152Perf <- caret::confusionMatrix(
          as.factor(ResNet152PredictedClass),
          as.factor(outputTableResNet152Sub$ActualClass),
          mode = 'everything'
        )$byClass

        TempRowResNet152 <- cbind.data.frame(
          t(ResNet152Perf),
          ResNet152.loss,
          trainingfolder,
          n.epoch,
          'ResNet152'
        )

        colnames(TempRowResNet152) <- c(
          "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
          "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
          "Detection Prevalence", "Balanced Accuracy",
          "Validation loss",
          "Training Data",
          "N epochs",
          "CNN Architecture"
        )

        ROCRpred <- ROCR::prediction(predictions = outputTableResNet152Sub$Probability, labels = outputTableResNet152Sub$ActualClass)
        AUCval <- ROCR::performance(ROCRpred, 'auc')
        TempRowResNet152$AUC <- AUCval@y.values[[1]]
        TempRowResNet152$Threshold <- as.character(threshold)
        TempRowResNet152$Frozen <- unfreeze
        TempRowResNet152$Class <- UniqueClasses[b]
        TempRowResNet152$Class <- as.factor(TempRowResNet152$Class)
        CombinedTempRow <- rbind.data.frame(CombinedTempRow, TempRowResNet152)
      }
    }

    filename <- paste(output.base.path, 'performance_tables/', trainingfolder, '_', n.epoch, '_', '_TransferLearningCNNDFResNet152.csv', sep = '')
    write.csv(CombinedTempRow, filename, row.names = FALSE)

    filename_multi <- paste(output.base.path, 'performance_tables_multi/', trainingfolder, '_', n.epoch, '_', '_TransferLearningCNNDFResNet152multi.csv', sep = '')

    ResNet152Perf <- caret::confusionMatrix(
      as.factor(outputTableResNet152$PredictedClass),
      as.factor(outputTableResNet152$ActualClass),
      mode = 'everything'
    )$byClass

    TempRowResNet152 <- cbind.data.frame(
      (ResNet152Perf),
      ResNet152.loss,
      trainingfolder,
      n.epoch,
      'ResNet152'
    )

    colnames(TempRowResNet152) <- c(
      "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
      "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
      "Detection Prevalence", "Balanced Accuracy",
      "Validation loss",
      "Training Data",
      "N epochs",
      "CNN Architecture"
    )

    TempRowResNet152$Class <- str_split_fixed(rownames(TempRowResNet152), pattern = ': ', n = 2)[, 2]

    write.csv(TempRowResNet152, filename_multi, row.names = FALSE)

    rm(modelResNet152Gibbon)
  }
}
