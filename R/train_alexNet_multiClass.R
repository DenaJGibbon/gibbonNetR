#' Train a Multi-Class AlexNet Model
#'
#' This function trains an AlexNet model on a given dataset for multi-class image classification.
#' The trained model, along with performance metrics and other metadata, are saved to disk.
#'
#' @param input.data.path Character. The path to the input training data folder. Sub-folders should correspond to class labels.
#' @param test.data Character. The path to the input test data folder. Sub-folders should correspond to class labels.
#' @param unfreeze Logical. If TRUE, all layers of the pretrained AlexNet will be unfrozen for retraining. Default is TRUE.
#' @param epoch.iterations Integer. The number of epochs to train the model for. Default is 1.
#' @param early.stop Character. If "yes", early stopping will be applied during training. Default is "yes".
#' @param output.base.path Character. The base path where output files will be saved. Default is 'data/'.
#' @param trainingfolder Character. A descriptor for naming the output folder.
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
#'   train_alexNet_multiClass(
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

train_alexNet_multiClass <- function(input.data.path, test.data, unfreeze = TRUE,
                                     epoch.iterations=1, early.stop = 'yes',
                                     output.base.path = 'data/',
                                     trainingfolder) {

  device <- if (cuda_is_available()) "cuda" else "cpu"

  to_device <- function(x, device) {
    x$to(device = device)
  }

  output.data.path <- paste(output.base.path, 'output', 'unfrozen', unfreeze, trainingfolder, '/', sep = '_')
  dir.create(output.data.path, showWarnings = FALSE)

  metadata <- tibble(
    Model_Name = "alexNet",
    Training_Data_Path = input.data.path,
    Test_Data_Path = test.data,
    Output_Path = output.data.path,
    Device_Used = device,
    EarlyStop = early.stop,
    Layers_Ununfrozen = unfreeze,
    Epochs = epoch.iterations
  )

  write_csv(metadata, paste0(output.data.path, "alexNetmodel_metadata.csv"))

  # Data loaders setup
  train_ds <- image_folder_dataset(
    file.path(input.data.path,'train' ),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(size = c(224, 224)) %>%
      torchvision::transform_color_jitter() %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
  )

  valid_ds <- image_folder_dataset(
    file.path(input.data.path, "valid"),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(size = c(224, 224)) %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
  )

  train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
  valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)


  # Read class names
  class_names <- sort(unique(attr(train_ds$class_to_idx, "names")))
  num_classes <- length(class_names)
  print(paste('Detected classes:', paste(class_names, collapse = ', ')))

  for(a in 1:length(epoch.iterations )){
    print('Training alexNet')
    n.epoch <- epoch.iterations [a]

  # Define the model
  net <- nn_module(
    "AlexNet",
    initialize = function() {
      self$model <- torchvision::model_alexnet(pretrained = TRUE)

      if (unfreeze == FALSE) {
        for (param in self$model$parameters()) {
          param$requires_grad_(FALSE)
        }
      }

      self$model$classifier <- nn_sequential(
        nn_dropout(0.5),
        nn_linear(9216, 512),
        nn_relu(),
        nn_linear(512, 256),
        nn_relu(),
        nn_linear(256, num_classes)
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
        luz_metric_accuracy()
      )
    )

  # Training the model
  if (early.stop == 'yes') {

    modelalexNetGibbon <- fitted %>%
      fit(train_dl, epochs = n.epoch, valid_data = valid_dl,
          callbacks = list(
            luz_callback_early_stopping(patience = 2),
            luz_callback_lr_scheduler(
              lr_one_cycle,
              max_lr = 0.01,
              epochs = n.epoch,
              steps_per_epoch = length(train_dl),
              call_on = "on_batch_end"
            ),
            #luz_callback_model_checkpoint(path = paste(output.data.path, trainingfolder,'/', sep = '_')),
            luz_callback_csv_logger(paste(output.data.path, trainingfolder, n.epoch, "logs_alexNet.csv", sep = '_'))
          ),
          verbose = TRUE
      )
  } else {
    modelalexNetGibbon <- fitted %>%
      fit(train_dl, epochs = n.epoch, valid_data = valid_dl,
          callbacks = list(
           # luz_callback_early_stopping(patience = 2),
            luz_callback_lr_scheduler(
              lr_one_cycle,
              max_lr = 0.01,
              epochs = n.epoch,
              steps_per_epoch = length(train_dl),
              call_on = "on_batch_end"
            ),
            #luz_callback_model_checkpoint(path = paste(output.data.path, trainingfolder,'/', sep = '_')),
            luz_callback_csv_logger(paste(output.data.path, trainingfolder, n.epoch, "logs_alexNet.csv", sep = '_'))
          ),
          verbose = TRUE
      )
  }


  luz_save(modelalexNetGibbon, paste( output.data.path,trainingfolder,n.epoch, "modelalexNet.pt",sep='_'))

  TempCSV.alexNet <- read.csv(paste(output.data.path, trainingfolder, n.epoch, "logs_alexNet.csv", sep = '_'))
  alexNet.loss <- TempCSV.alexNet[nrow(TempCSV.alexNet),]$loss

  LossPlot <- ggline(data=TempCSV.alexNet,x='epoch',y='loss',color = 'set')
  print(LossPlot)

  # Calculate performance metrics for alexNet -------------------------------------
  dir.create(paste(output.data.path,'performance_tables',sep=''))

  test_ds <- image_folder_dataset(
    file.path(test.data, "test"),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(size = c(224, 224)) %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
  )

  test_dl <- dataloader(test_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)


  # Get the list of image files
  imageFiles <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = TRUE)
  imageFileShort <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = FALSE)
  Folder <- str_split_fixed(imageFileShort,pattern = '/',n=2)[,1]
  imageFileShort <- str_split_fixed(imageFileShort,pattern = '/',n=2)[,2]

  # Prepare output tables
  outputTablealexNet <- data.frame()

  # Predict using alexNet
  alexNetPred <- predict(modelalexNetGibbon, test_dl)

  # Return the index of the max values (i.e. which class)
  PredMPS <- torch_argmax(alexNetPred, dim = 2)

  # Save to cpu
  PredMPS <- as_array(torch_tensor(PredMPS,device = 'cpu'))

  # Convert to a factor
  modelResnetGibbonPred <- as.factor(PredMPS)
  print(modelResnetGibbonPred)

  # Calculate the probability associated with each class
  Probability <- as_array(torch_tensor(nnf_softmax(alexNetPred, dim = 2),device = 'cpu'))
  #predictedResnet <- as.factor(ifelse(Probability[,1] > 0.5,1,2))

  # Find the index of the maximum value in each row
  max_prob_idx <- apply(Probability, 1, which.max)

  # Map the index to actual probability
  predicted_class_probability <- sapply(1:nrow(Probability), function(i) Probability[i, max_prob_idx[i]])

  # Convert the integer predictions to factor and then to character based on the levels
  modelResnetGibbonNames <- factor(modelResnetGibbonPred, levels = 1:length(class_names), labels = class_names)

  outputTablealexNet <- cbind.data.frame(modelResnetGibbonNames,predicted_class_probability)
  colnames(outputTablealexNet) <- c('PredictedClass','Probability')
  outputTablealexNet$ActualClass <- Folder

  # Save the output table as CSV file
  write.csv(outputTablealexNet, paste(output.data.path, trainingfolder, n.epoch, "output_alexNet.csv", sep = '_'), row.names = FALSE)

     alexNetPerf <- caret::confusionMatrix(
      as.factor(outputTablealexNet$PredictedClass),
      as.factor(outputTablealexNet$ActualClass),
      mode = 'everything'
    )$byClass

    TempRowalexNet <- cbind.data.frame(
      (alexNetPerf),
      alexNet.loss,
      trainingfolder,
      n.epoch,
      'alexNet'
    )

    colnames(TempRowalexNet) <- c(
      "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
      "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
      "Detection Prevalence", "Balanced Accuracy",
      "Validation loss",
      "Training Data",
      "N epochs",
      "CNN Architecture"
    )


  TempRowalexNet$Frozen <- unfreeze.param
  filename <- paste(output.data.path,'performance_tables/', trainingfolder, '_', n.epoch, '_', '_TransferLearningCNNDFalexNet.csv', sep = '')
  write.csv(TempRowalexNet, filename, row.names = FALSE)

  rm(modelalexNetGibbon)
}
}
