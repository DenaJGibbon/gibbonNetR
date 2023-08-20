#' Train the ResNet50 Model
#'
#' This function is designed to train the ResNet50 model on a given dataset.
#' The model is saved, and other metadata is stored for further usage.
#'
#' @param input.data.path Character. The path to the input data folder.
#' @param test.data Character. The path to the test data folder.
#' @param unfreeze Logical. Determines if all layers of the pretrained ResNet50 should be unfrozen for retraining.
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
#'   train_ResNet50(
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

train_ResNet50 <- function(input.data.path, test.data, unfreeze = TRUE,
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
    Model_Name = "ResNet50",
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

  write_csv(metadata, paste0(output.data.path, "ResNet50model_metadata.csv"))

  for(a in 1:length(epoch.iterations )){
    print('Training ResNet50')
    n.epoch <- epoch.iterations [a]

    # Data loaders setup
    train_ds <- image_folder_dataset(
      file.path(input.data.path,'train' ),
      transform = . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_color_jitter() %>%
        transform_resize(256) %>%
        transform_center_crop(224) %>%
        transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)), target_transform = function(x) as.double(x) - 1 )

    valid_ds <- image_folder_dataset(
      file.path(input.data.path, "valid"),
      transform = . %>%
        torchvision::transform_to_tensor() %>%
        transform_resize(256) %>%
        transform_center_crop(224) %>%
        transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)), target_transform = function(x) as.double(x) - 1 )


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

    net <- nn_module(
      initialize = function() {
        self$model <- model_resnet50(pretrained = TRUE)
        for (par in self$parameters) {
          par$requires_grad_(unfreeze.param) # False means the features are unfrozen
        }
        self$model$fc <- nn_sequential(
          nn_linear(self$model$fc$in_features, 1024),
          nn_relu(),
          nn_linear(1024, 1024),
          nn_relu(),
          nn_linear(1024, 1)
        )
      },
      forward = function(x) {
        output <- self$model(x)
        torch_squeeze(output, dim=2)
      }
    )

    model <- net %>%
      luz::setup(
        loss = nn_bce_with_logits_loss(),
        optimizer = optim_adam,
        metrics = list(
          luz_metric_binary_accuracy_with_logits()
        )
      )


    # rates_and_losses <- model %>% lr_finder(train_dl)
    # rates_and_losses %>% plot()

    modelResNet50Gibbon <- model %>%
      fit(train_dl, epochs=n.epoch, valid_data = valid_dl,
          callbacks = list(
            luz_callback_early_stopping(patience = 2),
            luz_callback_lr_scheduler(
              lr_one_cycle,
              max_lr = 0.01,
              epochs=n.epoch,
              steps_per_epoch = length(train_dl),
              call_on = "on_batch_end"),
            #luz_callback_model_checkpoint(path = "output_unfrozenbin_trainaddedclean/"),
            luz_callback_csv_logger(paste( output.data.path,trainingfolder,n.epoch, "logs_ResNet50.csv",sep='_'))
          ),
          verbose = TRUE)

    luz_save(modelResNet50Gibbon, paste( output.data.path,trainingfolder,n.epoch, "modelResNet50.pt",sep='_'))

    TempCSV.ResNet50 <- read.csv(paste(output.data.path, trainingfolder, n.epoch, "logs_ResNet50.csv", sep = '_'))
    ResNet50.loss <- TempCSV.ResNet50[nrow(TempCSV.ResNet50),]$loss

    LossPlot <- ggline(data=TempCSV.ResNet50,x='epoch',y='loss',color = 'set')
    print(LossPlot)

    # Calculate performance metrics for ResNet50 -------------------------------------
    dir.create(paste(output.data.path,'performance_tables',sep=''))

    # Get the list of image files
    imageFiles <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = TRUE)
    imageFileShort <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = FALSE)
    Folder <- str_split_fixed(imageFileShort,pattern = '/',n=2)[,1]
    imageFileShort <- str_split_fixed(imageFileShort,pattern = '/',n=2)[,2]

    # Prepare output tables
    outputTableResNet50 <- data.frame()

    test_ds <- image_folder_dataset(
      file.path(test.data, "test/"),
      transform = . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_resize(size = c(224, 224)) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
      target_transform = function(x) as.double(x) - 1
    )

    test_dl <- dataloader(test_ds, batch_size = 32, shuffle = F)

    # Predict using ResNet50
    ResNet50Pred <- predict(modelResNet50Gibbon, test_dl)
    ResNet50Prob <- torch_sigmoid(ResNet50Pred)
    ResNet50Prob <- as_array(torch_tensor(ResNet50Prob, device = 'cpu'))
    ResNet50Class <- ifelse((ResNet50Prob) < 0.5, positive.class, negative.class)

    # Add the results to output tables
    outputTableResNet50 <- rbind(outputTableResNet50, data.frame(Label = Folder, Probability = ResNet50Prob, PredictedClass = ResNet50Class, ActualClass = Folder))

    # Save the output table as CSV file
    write.csv(outputTableResNet50, paste(output.data.path, trainingfolder, n.epoch, "output_ResNet50.csv", sep = '_'), row.names = FALSE)

    # Initialize data frames
    CombinedTempRow <- data.frame()
    TransferLearningCNNDF <- data.frame()
    thresholds <- seq(0.1,1,0.1)

    for (threshold in thresholds) {
      # ResNet50
      ResNet50PredictedClass <- ifelse((outputTableResNet50$Probability) < threshold, positive.class, negative.class)

      ResNet50Perf <- caret::confusionMatrix(
        as.factor(ResNet50PredictedClass),
        as.factor(outputTableResNet50$ActualClass),
        mode = 'everything'
      )$byClass

      TempRowResNet50 <- cbind.data.frame(
        t(ResNet50Perf),
        ResNet50.loss,
        trainingfolder,
        n.epoch,
        'ResNet50'
      )

      colnames(TempRowResNet50) <- c(
        "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
        "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
        "Detection Prevalence", "Balanced Accuracy",
        "Validation loss",
        "Training Data",
        "N epochs",
        "CNN Architecture"
      )

      TempRowResNet50$Threshold <- as.character(threshold)
      CombinedTempRow <- rbind.data.frame(CombinedTempRow, TempRowResNet50)
    }

    TransferLearningCNNDF <- rbind.data.frame(TransferLearningCNNDF, CombinedTempRow)
    TransferLearningCNNDF$Frozen <- unfreeze.param
    filename <- paste(output.data.path,'performance_tables/', trainingfolder, '_', n.epoch, '_', '_TransferLearningCNNDFResNet50.csv', sep = '')
    write.csv(TransferLearningCNNDF, filename, row.names = FALSE)

    rm(modelResNet50Gibbon)
  }
}
