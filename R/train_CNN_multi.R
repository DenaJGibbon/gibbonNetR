#' Train Multi-class pretrained CNN Models
#'
#' This function facilitates training of convolutional neural network (CNN) models using various transfer learning architectures such as AlexNet, VGG16, VGG19, ResNet18, ResNet50, or ResNet152, on a given dataset. The trained model is saved along with metadata for further usage.
#'
#' @param input.data.path Character. Path to the input data folder.
#' @param test.data Character. Path to the test data folder.
#' @param architecture Character. Specifies the CNN architecture to use ('alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152').
#' @param unfreeze.param Logical. Indicates whether all layers of the pretrained CNN should be unfrozen for retraining. Default is TRUE.
#' @param batch_size Numeric. Batch size for training the model. Default is 32.
#' @param learning_rate Numeric. Learning rate for training the model.
#' @param save.model Logical. Specifies whether to save the trained model for future use. Default is FALSE.
#' @param class_weights Numeric vector. Weights assigned to different classes for handling class imbalance. Default is c(0.49, 0.49, 0.02).
#' @param epoch.iterations List of integers. Number of epochs for training the model. Default is 1.
#' @param early.stop Character. Indicates whether early stopping should be applied or not. Use "yes" to apply and "no" to skip. Default is 'yes'.
#' @param output.base.path Character. Base path where the output files should be saved. Default is 'data/'.
#' @param trainingfolder Character. A descriptive name for the training data, used for naming output files.
#' @param noise.category Character. Label for the noise category. Default is "Noise".
#' @param brightness Numeric. Brightness adjustment factor for color jitter. A value of 0 means no change. Higher values increase brightness. Default is 0.
#' @param contrast Numeric. Contrast adjustment factor for color jitter. A value of 0 means no change. Higher values increase contrast. Default is 0.
#' @param saturation Numeric. Saturation adjustment factor for color jitter. A value of 0 means no change. Higher values increase color saturation. Default is 0.
#' @return The function generates multiple output files, including:
#' \itemize{
#'   \item \strong{Trained Models}: If save.model = TRUE outputs saved model files (.pt) for specified architectures.
#'   \item \strong{Training Logs}: logs_model.csv containing logs of training sessions, including loss and accuracy metrics.
#'   \item \strong{Metadata}: model_metadata.csv contains metadata from training run
#'   \item \strong{Model predictions}: Saved for each architecture in output_TrainedModel_testdata.csv
#'   \item \strong{Performance Evaluation}: Saves .csv summarizing performance for each architecture in nested folder in output.base.path 'performance_tables_multi'.
#' }

#'
#' @examples
#' {{ input.data.path <- system.file("extdata", "multiclass/", package = "gibbonNetR")
#'   test.data <- system.file("extdata", "multiclass/test/", package = "gibbonNetR")
#'   result <- train_CNN_multi(
#'     input.data.path = input.data.path,
#'     test.data = test.data,
#'     architecture = "alexnet", # Choose architecture
#'     unfreeze.param = TRUE,
#'     class_weights = rep((1 / 5), 5),
#'     batch_size = 6,
#'     learning_rate = 0.001,
#'     epoch.iterations = 1, # Or any other list of integer epochs
#'     early.stop = "yes",
#'     output.base.path = paste(tempdir(), "/", sep = ""),
#'     trainingfolder = "test",
#'     noise.category = "noise"
#'   )
#'   print(result) }}

#' @seealso \code{\link[torch]{nn_module}} and other torch functions.
#' @importFrom stringr str_replace str_split_fixed
#' @importFrom tibble tibble
#' @importFrom readr write_csv
#' @importFrom magrittr %>%
#' @importFrom ggpubr ggline
#' @importFrom utils write.csv read.csv
#' @importFrom ROCR prediction performance
#' @importFrom tidyr pivot_longer
#' @note Requires train, valid, and test folders
#' created using created using 'spectrogram_images'
#' @export
#'
train_CNN_multi <- function(input.data.path, test.data, architecture,
                            unfreeze.param = TRUE, batch_size = 32, learning_rate,
                            save.model = FALSE,
                            class_weights = c(0.49, 0.49, 0.02),
                            epoch.iterations = 1, early.stop = "yes",
                            output.base.path = tempdir(),
                            brightness = 0,
                            contrast = 0 ,
                            saturation = 0,
                            trainingfolder,
                            noise.category = "Noise") {
  # Device
  device <- if (cuda_is_available()) "cuda" else "cpu"

  to_device <- function(x, device) {
    x$to(device = device)
  }

  # Location to save the output
  output.data.path <- paste(output.base.path, trainingfolder, "multi", "unfrozen", unfreeze.param, "/", sep = "_")

  # Create if doesn't exist
  dir.create(output.data.path, showWarnings = FALSE, recursive = TRUE)

  # Metadata
  metadata <- data.frame(
    Model_Name = architecture,
    Training_Data_Path = input.data.path,
    Test_Data_Path = test.data,
    Output_Path = output.data.path,
    Device_Used = device,
    EarlyStop = early.stop,
    Layers_Unfrozen = unfreeze.param,
    Epochs = epoch.iterations,
    Learning_rate = learning_rate,
    Noise.class = noise.category
  )

  write.csv(metadata, paste0(output.data.path, architecture, "_model_metadata.csv"))

  for (a in 1:length(epoch.iterations)) {
    message(paste("Training", architecture))
    n.epoch <- epoch.iterations[a]


    if (architecture %in% c("alexnet", "vgg16", "vgg19") == TRUE) {
      # Data loaders setup
      train_ds <- image_folder_dataset(
        file.path(input.data.path, "train"),
        transform = . %>%
          torchvision::transform_to_tensor() %>%
          torchvision::transform_resize(size = c(224, 224)) %>%
          torchvision::transform_color_jitter(brightness = brightness,
                                              contrast = contrast ,
                                              saturation = saturation ) %>%
          torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
      )

      valid_ds <- image_folder_dataset(
        file.path(input.data.path, "valid"),
        transform = . %>%
          torchvision::transform_to_tensor() %>%
          torchvision::transform_resize(size = c(224, 224)) %>%
          torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
      )
    }

    if (architecture %in% c("resnet18", "resnet50", "resnet152") == TRUE) {
      train_ds <- image_folder_dataset(
        file.path(input.data.path, "train"),
        transform = . %>%
          torchvision::transform_to_tensor() %>%
          torchvision::transform_color_jitter(brightness = brightness,
                                              contrast = contrast ,
                                              saturation = saturation) %>%
          transform_resize(256) %>%
          transform_center_crop(224) %>%
          transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
      )

      valid_ds <- image_folder_dataset(
        file.path(input.data.path, "valid"),
        transform = . %>%
          torchvision::transform_to_tensor() %>%
          transform_resize(256) %>%
          transform_center_crop(224) %>%
          transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
      )
    }


    train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE, drop_last = TRUE)
    valid_dl <- dataloader(valid_ds, batch_size = batch_size, shuffle = FALSE, drop_last = TRUE)

    # Read class names
    class_names <- sort(unique(attr(train_ds$class_to_idx, "names")))
    num_classes <- length(class_names)
    cat("Detected classes:", paste(class_names, collapse = ", "), "\n")


    # Model setup
    if (architecture == "alexnet") {
      net <- torch::nn_module(
        initialize = function() {
          self$model <- model_alexnet(pretrained = TRUE)
          for (par in self$parameters) {
            par$requires_grad_(unfreeze.param)
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
          output <- self$model(x)
          torch_squeeze(output, dim = 2)
        }
      )
    } else if (architecture == "vgg16") {
      net <- torch::nn_module(
        initialize = function() {
          self$model <- model_vgg16(pretrained = TRUE)

          for (par in self$parameters) {
            par$requires_grad_(unfreeze.param)
          }

          self$model$classifier <- nn_sequential(
            nn_dropout(0.5),
            nn_linear(25088, 4096),
            nn_relu(),
            nn_dropout(0.5),
            nn_linear(4096, 4096),
            nn_relu(),
            nn_linear(4096, num_classes)
          )
        },
        forward = function(x) {
          output <- self$model(x)
          torch_squeeze(output, dim = 2)
        }
      )
    } else if (architecture == "vgg19") {
      net <- torch::nn_module(
        initialize = function() {
          self$model <- model_vgg19(pretrained = TRUE)

          for (par in self$parameters) {
            par$requires_grad_(unfreeze.param)
          }

          self$model$classifier <- nn_sequential(
            nn_dropout(0.5),
            nn_linear(25088, 4096),
            nn_relu(),
            nn_dropout(0.5),
            nn_linear(4096, 4096),
            nn_relu(),
            nn_linear(4096, num_classes)
          )
        },
        forward = function(x) {
          output <- self$model(x)
          torch_squeeze(output, dim = 2)
        }
      )
    } else if (architecture == "resnet18") {
      net <- nn_module(
        initialize = function() {
          self$model <- model_resnet18(pretrained = TRUE)
          for (par in self$parameters) {
            par$requires_grad_(unfreeze.param) #
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
          output <- self$model(x)
          torch_squeeze(output, dim = 2)
        }
      )
    } else if (architecture == "resnet50") {
      net <- nn_module(
        initialize = function() {
          self$model <- model_resnet50(pretrained = TRUE)
          for (par in self$parameters) {
            par$requires_grad_(unfreeze.param) #
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
          output <- self$model(x)
          torch_squeeze(output, dim = 2)
        }
      )
    } else if (architecture == "resnet152") {
      net <- nn_module(
        initialize = function() {
          self$model <- model_resnet152(pretrained = TRUE)
          for (par in self$parameters) {
            par$requires_grad_(unfreeze.param) #
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
          output <- self$model(x)
          torch_squeeze(output, dim = 2)
        }
      )
    } else {
      stop("Invalid architecture specified. Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'.")
    }

    weight <- torch_tensor(class_weights, device = "cpu")

    fitted <- net %>%
      luz::setup(
        loss = nn_cross_entropy_loss(weight = weight),
        optimizer = optim_adam,
        metrics = list(
          luz_metric_accuracy()
        )
      )

    # Training the model
    if (early.stop == "yes") {
      multiModel <- fitted %>%
        fit(train_dl,
          epochs = n.epoch, valid_data = valid_dl,
          callbacks = list(
            luz_callback_early_stopping(patience = 2),
            luz_callback_lr_scheduler(
              lr_one_cycle,
              max_lr = learning_rate,
              epochs = n.epoch,
              steps_per_epoch = length(train_dl),
              call_on = "on_batch_end"
            ),
            luz_callback_csv_logger(paste(output.data.path, trainingfolder, n.epoch, architecture, "logs_model.csv", sep = "_"))
          ),
          verbose = TRUE,
          accelerator = accelerator(cpu = TRUE)
        )
    } else {
      multiModel <- fitted %>%
        fit(train_dl,
          epochs = n.epoch, valid_data = valid_dl,
          callbacks = list(
            luz_callback_lr_scheduler(
              lr_one_cycle,
              max_lr = learning_rate,
              epochs = n.epoch,
              steps_per_epoch = length(train_dl),
              call_on = "on_batch_end"
            ),
            luz_callback_csv_logger(paste(output.data.path, trainingfolder, n.epoch, architecture, "logs_model.csv", sep = "_"))
          ),
          verbose = TRUE,
          accelerator = accelerator(cpu = TRUE)
        )
    }


    if (save.model == TRUE) {
      luz_save(multiModel, paste(output.data.path, trainingfolder, n.epoch, architecture, "model.pt", sep = "_"))
    }


    TempCSV.TrainedModel <- read.csv(paste(output.data.path, trainingfolder, n.epoch, architecture, "logs_model.csv", sep = "_"))
    TrainedModel.loss <- TempCSV.TrainedModel[nrow(TempCSV.TrainedModel), ]$loss


    LossPlot <- ggline(data = TempCSV.TrainedModel, x = "epoch", y = "loss", color = "set")
    print(LossPlot)

    # Get the list of image files
    imageFiles <- list.files(paste(test.data, sep = ""), recursive = TRUE, full.names = TRUE)
    imageFileShort <- list.files(paste(test.data, sep = ""), recursive = TRUE, full.names = FALSE)
    Folder <- str_split_fixed(imageFileShort, pattern = "/", n = 2)[, 1]
    imageFileShort <- str_split_fixed(imageFileShort, pattern = "/", n = 2)[, 2]

    # Prepare output tables
    outputTableTrainedModel <- data.frame()

    # Define transforms based on model type
    if (str_detect(architecture, pattern = "resnet")) {
      transform_list <- . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_color_jitter(brightness = brightness,
                                            contrast = contrast ,
                                            saturation = saturation) %>%
        transform_resize(256) %>%
        transform_center_crop(224) %>%
        transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    } else {
      transform_list <- . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_resize(size = c(224, 224)) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }

    test_ds <- image_folder_dataset(test.data, transform = transform_list)
    test_dl <- dataloader(test_ds, batch_size = batch_size, shuffle = FALSE)

    # Predict using TrainedModel
    TrainedModelPred <- predict(multiModel, test_dl)

    # Return the index of the max values (i.e. which class)
    PredMPS <- torch_argmax(TrainedModelPred, dim = 2)

    # Save to cpu
    PredMPS <- as_array(torch_tensor(PredMPS, device = "cpu"))

    # Convert to a factor
    modelMultiPred <- as.factor(PredMPS)

    # Calculate the probability associated with each class
    Probability <- as_array(torch_tensor(nnf_softmax(TrainedModelPred, dim = 2), device = "cpu"))

    # Find the index of the maximum value in each row
    max_prob_idx <- apply(Probability, 1, which.max)

    # Map the index to actual probability
    predicted_class_probability <- sapply(1:nrow(Probability), function(i) Probability[i, max_prob_idx[i]])

    # Convert the integer predictions to factor and then to character based on the levels
    modelMultiNames <- factor(modelMultiPred, levels = 1:length(class_names), labels = class_names)

    outputTableMulti <- cbind.data.frame(Probability,Folder)
    colnames(outputTableMulti) <- c(class_names,"ActualClass" )

    # Save the output table as CSV file
    write.csv(
      outputTableMulti,
      paste(
        output.data.path,
        trainingfolder, n.epoch, architecture,
        "output_TrainedModel_testdata.csv",
        sep = "_"
      ),
      row.names = FALSE
    )

    UniqueClasses <- class_names
    UniqueClasses <- UniqueClasses[-which(UniqueClasses == noise.category)]

    # Initialize data frames
    CombinedTempRow <- data.frame()
    TransferLearningCNNDF <- data.frame()
    thresholds <- seq(0.1, 1, 0.1)

    for (b in 1:length(UniqueClasses)) {
      current_class <- UniqueClasses[b]

      outputTableMultiSub <- outputTableMulti

      # Subset to relevant rows (only current class and noise)

      binary_subset <- subset(outputTableMultiSub,
                              ActualClass== current_class |
                                ActualClass== noise.category )

      binary_subset$binary_label <-
        ifelse(binary_subset$ActualClass==current_class,1,0)

      binary_subset$Probability <- binary_subset[,current_class]

      if (sum(binary_subset$binary_label) == 0) {
        message(paste("Skipping", current_class, "- cannot calculate performance"))
      } else {
        pred <- prediction(binary_subset$Probability, binary_subset$binary_label)
        AUCval <- performance(pred, measure = "auc")

        for (threshold in thresholds) {
          predicted_class <- ifelse(binary_subset$Probability > threshold,
                                    current_class, noise.category)

          predicted_class <- factor(predicted_class,
                                    levels = levels(as.factor(binary_subset$ActualClass)))

          cm <- caret::confusionMatrix(
            predicted_class,
            as.factor(binary_subset$ActualClass),
            mode = "everything"
          )$byClass

          TempRowMulti <- cbind.data.frame(
            t(cm),
            TrainedModel.loss,
            trainingfolder,
            n.epoch,
            architecture
          )


          colnames(TempRowMulti) <- c(
            "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
            "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
            "Detection Prevalence", "Balanced Accuracy",
            "Validation loss",
            "Training Data",
            "N epochs",
            "CNN Architecture"
          )

          TempRowMulti$AUC <- as.numeric(AUCval@y.values)
          TempRowMulti$Threshold <- as.character(threshold)
          TempRowMulti$Frozen <- unfreeze.param
          TempRowMulti$Class <- UniqueClasses[b]
          TempRowMulti$Class <- as.factor(TempRowMulti$Class)
          TempRowMulti$TestDataPath <- test.data
          CombinedTempRow <- rbind.data.frame(CombinedTempRow, TempRowMulti)
        }

      }
    }

    # Location to save the output
    output.data.performance <- paste(output.data.path, "performance_tables_multi/", sep = "")

    message(output.data.performance)

    output.data.performance <- paste(output.data.path, "performance_tables_multi/", sep = "")

    # Create if doesn't exist
    dir.create(output.data.performance, showWarnings = FALSE, recursive = T)

    filename <- paste(output.data.performance, trainingfolder, "_", n.epoch, "_", architecture, "_TransferLearningCNNDFMultiThreshold.csv", sep = "")
    write.csv(CombinedTempRow, filename, row.names = FALSE)

    rm(multiModel)
  }
}
