#' Train the AlexNet Model
#'
#' This function is designed to train the AlexNet model on a given dataset.
#' The model is saved, and other metadata is stored for further usage.
#'
#' @param input.data.path Character. The path to the input data folder.
#' @param test.data Character. The path to the test data folder.
#' @param unfreeze Logical. Determines if all layers of the pretrained AlexNet should be unfrozen for retraining.
#'                 Default is TRUE.
#' @param epoch.iterations List of integers. The number of epochs for training the model. Default is 20.
#' @param early.stop Character. Determines whether early stopping should be applied or not.
#'                   "yes" to apply and "no" to skip. Default is 'yes'.
#' @param output.base.path Character. The base path where the output files should be saved.
#'                          Default is 'data/'.
#' @param trainingfolder Character. A shortened descriptor of the training data, used for naming output files.
#'                             Default is 'imagesmalaysiaHQ'.
#' @param positive.class Character. The name of the positive class label. Default is 'Gibbons'.
#' @param negative.class Character. The name of the negative class label. Default is 'Noise'.
#'
#' @return A list containing two elements:
#' \itemize{
#'   \item \strong{Output_Path}: The path where the model and metadata are saved.
#'   \item \strong{Metadata}: A dataframe containing metadata about the training session.
#' }
#'
#' @examples
#' \dontrun{
#'   train_alexnet(
#'     input.data.path = "path_to_input_data",
#'     test.data = "path_to_test_data",
#'     unfreeze = TRUE,
#'     epoch.iterations = list(20),  # Or any other list of integer epochs
#'     early.stop = "yes",
#'     output.base.path = "data/",
#'     trainingfolder = "example_folder_name"
#'   )
#' }
#' @seealso \code{\link[torch]{nn_module}} and other torch functions.
#' @export
#' @importFrom stringr str_replace str_split_fixed
#' @importFrom tibble tibble
#' @importFrom readr write_csv
#' @importFrom magrittr %>%
#' @importFrom ggpubr ggline
#'

train_alexnet <- function(input.data.path, test.data, unfreeze = TRUE,
                          epoch.iterations, early.stop = 'yes',
                          output.base.path = 'data/',
                          trainingfolder,
                          positive.class="Gibbons",
                          negative.class="Noise") {

  # Device
  device <- if(cuda_is_available()) "cuda" else "cpu"

  to_device <- function(x, device) {
    x$to(device = device)
  }

  # Location to save the output
  output.data.path <- paste(output.base.path,'output','unfrozen', unfreeze, trainingfolder, '/', sep='_')

  # Create if doesn't exist
  dir.create(output.data.path, showWarnings = FALSE)

  # Metadata
  metadata <- tibble(
    Model_Name = "AlexNet",
    Training_Data_Path = input.data.path,
    Test_Data_Path = test.data,
    Output_Path = output.data.path,
    Device_Used = device,
    EarlyStop = early.stop,
    Layers_Ununfrozen = unfreeze,
    Epochs = epoch.iterations,
    Positive.class=positive.class,
    Negative.class=negative.class
  )

  write_csv(metadata, paste0(output.data.path, "AlexNetmodel_metadata.csv"))

for(a in 1:length(epoch.iterations )){
print('Training AlexNet')
  n.epoch <- epoch.iterations [a]

    # Data loaders setup
    train_ds <- image_folder_dataset(
      file.path(input.data.path,'train' ),
      transform = . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_resize(size = c(224, 224)) %>%
        torchvision::transform_color_jitter() %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
      target_transform = function(x) as.double(x) - 1
    )

    valid_ds <- image_folder_dataset(
      file.path(input.data.path, "valid"),
      transform = . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_resize(size = c(224, 224)) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
      target_transform = function(x) as.double(x) - 1
    )


   TrainingLabelsMatch <- attr(train_ds$class_to_idx, "names")[1] == c(positive.class) &
     attr(train_ds$class_to_idx, "names")[2] == c(negative.class)

    if( TrainingLabelsMatch == FALSE) {
      print( 'Training classes do not match! This is the order of the classes based on training folder')
      print(train_ds$class_to_idx)
      break

    } else {

      print(paste('Postive class =', positive.class,' and Negative class =', negative.class))

    }


    train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
    valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)

    # AlexNet Training
    net <- torch::nn_module(
      initialize = function() {
        self$model <- model_alexnet(pretrained = TRUE)
        for (par in self$parameters) {
          par$requires_grad_(unfreeze)
        }

        self$model$classifier <- nn_sequential(
          nn_dropout(0.5),
          nn_linear(9216, 512),
          nn_relu(),
          nn_linear(512, 256),
          nn_relu(),
          nn_linear(256, 1)
        )
      },
      forward = function(x) {
        output <- self$model(x)
        torch_squeeze(output, dim=2)
      }
    )

    fitted <- net %>%
      luz::setup(
        loss = nn_bce_with_logits_loss(),
        optimizer = optim_adam,
        metrics = list(
          luz_metric_binary_accuracy_with_logits()
        )
      )

    modelAlexNetGibbon <- fitted %>%
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
            luz_callback_csv_logger(paste(output.data.path, trainingfolder, n.epoch, "logs_AlexNet.csv", sep = '_'))
          ),
          verbose = TRUE
      )

    luz_save(modelAlexNetGibbon, paste( output.data.path,trainingfolder,n.epoch, "modelAlexNet.pt",sep='_'))

    TempCSV.AlexNet <- read.csv(paste(output.data.path, trainingfolder, n.epoch, "logs_AlexNet.csv", sep = '_'))
    AlexNet.loss <- TempCSV.AlexNet[nrow(TempCSV.AlexNet),]$loss

    LossPlot <- ggline(data=TempCSV.AlexNet,x='epoch',y='loss',color = 'set')
    print(LossPlot)

    # Calculate performance metrics for AlexNet -------------------------------------
    dir.create(paste(output.data.path,'performance_tables',sep=''))

    # Get the list of image files
    imageFiles <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = TRUE)
    imageFileShort <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = FALSE)
    Folder <- str_split_fixed(imageFileShort,pattern = '/',n=2)[,1]
    imageFileShort <- str_split_fixed(imageFileShort,pattern = '/',n=2)[,2]

    # Prepare output tables
    outputTableAlexNet <- data.frame()

    # Prepare dataset for prediction
    test_ds <- image_folder_dataset(
      file.path(test.data, "test/"),
      transform = . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_resize(size = c(224, 224)) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
      target_transform = function(x) as.double(x) - 1
    )

    test_dl <- dataloader(test_ds, batch_size = 32, shuffle = F)

    # Predict using AlexNet
    AlexNetPred <- predict(modelAlexNetGibbon, test_dl)
    AlexNetProb <- torch_sigmoid(AlexNetPred)
    AlexNetProb <- as_array(torch_tensor(AlexNetProb, device = 'cpu'))
    AlexNetClass <- ifelse((AlexNetProb) < 0.5, positive.class, negative.class)

    # Add the results to output tables
    outputTableAlexNet <- rbind(outputTableAlexNet, data.frame(Label = Folder, Probability = AlexNetProb, PredictedClass = AlexNetClass, ActualClass = Folder))

    # Save the output table as CSV file
    write.csv(outputTableAlexNet, paste(output.data.path, trainingfolder, n.epoch, "output_AlexNet.csv", sep = '_'), row.names = FALSE)

    # Initialize data frames
    CombinedTempRow <- data.frame()
    TransferLearningCNNDF <- data.frame()
    thresholds <- seq(0.1,1,0.1)

    for (threshold in thresholds) {
      # AlexNet
      AlexNetPredictedClass <- ifelse((outputTableAlexNet$Probability) < threshold, positive.class, negative.class)

      AlexNetPerf <- caret::confusionMatrix(
        as.factor(AlexNetPredictedClass),
        as.factor(outputTableAlexNet$ActualClass),
        mode = 'everything'
      )$byClass

      TempRowAlexNet <- cbind.data.frame(
        t(AlexNetPerf),
        AlexNet.loss,
        trainingfolder,
        n.epoch,
        'AlexNet'
      )

      colnames(TempRowAlexNet) <- c(
        "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
        "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
        "Detection Prevalence", "Balanced Accuracy",
        "Validation loss",
        "Training Data",
        "N epochs",
        "CNN Architecture"
      )

      TempRowAlexNet$Threshold <- as.character(threshold)
      CombinedTempRow <- rbind.data.frame(CombinedTempRow, TempRowAlexNet)
    }

   ROCRpred <-  ROCR::prediction(predictions = outputTableAlexNet$Probability,
                     labels = outputTableAlexNet$ActualClass)
   AUCval <- ROCR::performance(ROCRpred,'auc')
   CombinedTempRow$AUC <- AUCval@y.values[[1]]


    TransferLearningCNNDF <- rbind.data.frame(TransferLearningCNNDF, CombinedTempRow)
    TransferLearningCNNDF$Frozen <- unfreeze.param
    filename <- paste(output.data.path,'performance_tables/', trainingfolder, '_', n.epoch, '_', '_TransferLearningCNNDFAlexNET.csv', sep = '')
    write.csv(TransferLearningCNNDF, filename, row.names = FALSE)

    rm(modelAlexNetGibbon)
}
}
