#' Train Binary CNN Models
#'
#' This function trains Convolutional Neural Network (CNN) models, such as AlexNet, VGG16, VGG19, ResNet18, ResNet50, or ResNet152, on a given dataset. The trained model is saved along with metadata for further usage.
#'
#' @param input.data.path Character. The path to the folder containing the training data.
#' @param test.data Character. The path to the folder containing the test data.
#' @param architecture Character. The CNN architecture to use ('alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152').
#' @param noise.weight Numeric. Assigned weight for the noise class. Default is 0.5.
#' @param unfreeze.param Logical. Determines whether to unfreeze all layers of the pretrained CNN for retraining. Default is TRUE.
#' @param batch_size Numeric. Batch size for training the model. Default is 32.
#' @param learning_rate Numeric. The learning rate for training the model.
#' @param epoch.iterations Numeric. The number of epochs for training the model. Default is 1.
#' @param early.stop Character. Determines whether early stopping should be applied or not. Options: "yes" or "no". Default is 'yes'.
#' @param output.base.path Character. The base path where the output files should be saved. Default is 'data/'.
#' @param save.model Logical. Whether to save the trained model for future use. Default is FALSE.
#' @param trainingfolder Character. A descriptor of the training data used for naming output files.
#' @param positive.class Character. The name of the positive class label. Default is 'Gibbons'.
#' @param negative.class Character. The name of the negative class label. Default is 'Noise'.
#' @param list.thresholds Numerical list indicating thresholds. Default is seq(0.1,1,.1).
#'
#' @return A list containing two elements:
#' \itemize{
#'   \item \strong{Output_Path}: The path where the model and metadata are saved.
#'   \item \strong{Metadata}: A dataframe containing metadata about the training session.
#' }
#'
#' @examples
#' {
#'   input.data.path <- system.file("extdata", "binary/", package = "gibbonNetR")
#'   test.data <- system.file("extdata", "binary/test/", package = "gibbonNetR")
#'   result <- train_CNN_binary(
#'     input.data.path = input.data.path,
#'     test.data = test.data,
#'     architecture = "alexnet", # Choose architecture
#'     unfreeze.param = TRUE,
#'     batch_size = 6,
#'     learning_rate = 0.001,
#'     epoch.iterations = 1, # Or any other list of integer epochs
#'     early.stop = "yes",
#'     output.base.path = paste(tempdir(), "/", sep = ""),
#'     trainingfolder = "test_binary"
#'   )
#'   print(result)
#' }
#'
#' @seealso \code{\link[torch]{nn_module}} and other torch functions.
#'
#' @importFrom stringr str_replace str_split_fixed
#' @importFrom tibble tibble
#' @importFrom readr write_csv
#' @importFrom magrittr %>%
#' @importFrom ggpubr ggline
#' @importFrom utils write.csv read.csv
#' @importFrom ROCR prediction performance
#' @note Requires train, valid, and test folders
#' created using created using 'spectrogram_images'
#' @export
train_CNN_binary <-
  function(input.data.path,
           test.data,
           architecture,
           noise.weight = 0.5,
           unfreeze.param = TRUE,
           batch_size = 32,
           learning_rate,
           save.model = FALSE,
           epoch.iterations = 1,
           early.stop = "yes",
           output.base.path = "data/",
           trainingfolder,
           list.thresholds = seq(0.1, 1, .1),
           positive.class = "Gibbons",
           negative.class = "Noise") {
    # Device
    device <- if (cuda_is_available()) {
      "cuda"
    } else {
      "cpu"
    }

    to_device <- function(x, device) {
      x$to(device = device)
    }

    # Location to save the output
    output.data.path <-
      paste(output.base.path,
        trainingfolder,
        "binary",
        "unfrozen",
        unfreeze.param,
        "/",
        sep = "_"
      )

    # Create if doesn't exist
    dir.create(output.data.path,
      showWarnings = FALSE,
      recursive = TRUE
    )

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
      Positive.class = positive.class,
      Negative.class = negative.class
    )

    write.csv(
      metadata,
      paste0(output.data.path, architecture, "_model_metadata.csv")
    )

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
            torchvision::transform_color_jitter() %>%
            torchvision::transform_normalize(
              mean = c(0.485, 0.456, 0.406),
              std = c(0.229, 0.224, 0.225)
            ),
          target_transform = function(x) {
            as.double(x) - 1
          }
        )

        valid_ds <- image_folder_dataset(
          file.path(input.data.path, "valid"),
          transform = . %>%
            torchvision::transform_to_tensor() %>%
            torchvision::transform_resize(size = c(224, 224)) %>%
            torchvision::transform_normalize(
              mean = c(0.485, 0.456, 0.406),
              std = c(0.229, 0.224, 0.225)
            ),
          target_transform = function(x) {
            as.double(x) - 1
          }
        )
      }

      if (architecture %in% c("resnet18", "resnet50", "resnet152") == TRUE) {
        train_ds <- image_folder_dataset(
          file.path(input.data.path, "train"),
          transform = . %>%
            torchvision::transform_to_tensor() %>%
            torchvision::transform_color_jitter() %>%
            transform_resize(256) %>%
            transform_center_crop(224) %>%
            transform_normalize(
              mean = c(0.485, 0.456, 0.406),
              std = c(0.229, 0.224, 0.225)
            ),
          target_transform = function(x) {
            as.double(x) - 1
          }
        )

        valid_ds <- image_folder_dataset(
          file.path(input.data.path, "valid"),
          transform = . %>%
            torchvision::transform_to_tensor() %>%
            transform_resize(256) %>%
            transform_center_crop(224) %>%
            transform_normalize(
              mean = c(0.485, 0.456, 0.406),
              std = c(0.229, 0.224, 0.225)
            ),
          target_transform = function(x) {
            as.double(x) - 1
          }
        )
      }


      TrainingLabelsMatch <-
        attr(train_ds$class_to_idx, "names")[1] == c(positive.class) &
          attr(train_ds$class_to_idx, "names")[2] == c(negative.class)

      if (TrainingLabelsMatch == FALSE) {
        message(
          "Training classes do not match! This is based on the order of the classes in the training folder"
        )
        message(train_ds$class_to_idx)
        stop("Stopping due to class mismatch.") # Stops execution of further code
      } else {
        message(paste(
          "Postive class =",
          positive.class,
          " and Negative class =",
          negative.class
        ))
      }


      train_dl <-
        dataloader(
          train_ds,
          batch_size = batch_size,
          shuffle = TRUE,
          drop_last = TRUE
        )
      valid_dl <-
        dataloader(
          valid_ds,
          batch_size = batch_size,
          shuffle = FALSE,
          drop_last = TRUE
        )


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
              nn_linear(256, 2)
            )
          },
          forward = function(x) {
            self$model(x)[, 1]
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
              nn_linear(4096, 2)
            )
          },
          forward = function(x) {
            self$model(x)[, 1]
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
              nn_linear(4096, 2)
            )
          },
          forward = function(x) {
            self$model(x)[, 1]
          }
        )
      } else if (architecture == "resnet18") {
        net <- nn_module(
          initialize = function() {
            self$model <- model_resnet18(pretrained = TRUE)
            for (par in self$parameters) {
              par$requires_grad_(unfreeze.param) # False means the features are unfrozen
            }
            self$model$fc <- nn_sequential(
              nn_linear(self$model$fc$in_features, 1024),
              nn_relu(),
              nn_linear(1024, 1024),
              nn_relu(),
              nn_linear(1024, 2)
            )
          },
          forward = function(x) {
            self$model(x)[, 1]
          }
        )
      } else if (architecture == "resnet50") {
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
              nn_linear(1024, 2)
            )
          },
          forward = function(x) {
            self$model(x)[, 1]
          }
        )
      } else if (architecture == "resnet152") {
        net <- nn_module(
          initialize = function() {
            self$model <- model_resnet152(pretrained = TRUE)
            for (par in self$parameters) {
              par$requires_grad_(unfreeze.param) # False means the features are unfrozen
            }
            self$model$fc <- nn_sequential(
              nn_linear(self$model$fc$in_features, 1024),
              nn_relu(),
              nn_linear(1024, 1024),
              nn_relu(),
              nn_linear(1024, 2)
            )
          },
          forward = function(x) {
            self$model(x)[, 1]
          }
        )
      } else {
        stop(
          "Invalid architecture specified. Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'."
        )
      }

      pos_weight <-
        torch_tensor(rep(noise.weight, batch_size), device = "cpu")

      fitted <- net %>%
        luz::setup(
          loss = nn_bce_with_logits_loss(pos_weight = pos_weight),
          optimizer = optim_adam,
          metrics = list(luz_metric_binary_accuracy_with_logits())
        )

      # Training the model
      if (early.stop == "yes") {
        BinaryModel <- fitted %>%
          fit(
            train_dl,
            epochs = n.epoch,
            valid_data = valid_dl,
            callbacks = list(
              luz_callback_early_stopping(patience = 2),
              luz_callback_lr_scheduler(
                lr_one_cycle,
                max_lr = learning_rate,
                epochs = n.epoch,
                steps_per_epoch = length(train_dl),
                call_on = "on_batch_end"
              ),
              luz_callback_csv_logger(
                paste(
                  output.data.path,
                  trainingfolder,
                  n.epoch,
                  architecture,
                  "logs_model.csv",
                  sep = "_"
                )
              )
            ),
            verbose = TRUE,
            accelerator = accelerator(cpu = TRUE)
          )
      } else {
        BinaryModel <- fitted %>%
          fit(
            train_dl,
            epochs = n.epoch,
            valid_data = valid_dl,
            callbacks = list(
              luz_callback_lr_scheduler(
                lr_one_cycle,
                max_lr = learning_rate,
                epochs = n.epoch,
                steps_per_epoch = length(train_dl),
                call_on = "on_batch_end"
              ),
              luz_callback_csv_logger(
                paste(
                  output.data.path,
                  trainingfolder,
                  n.epoch,
                  architecture,
                  "logs_model.csv",
                  sep = "_"
                )
              )
            ),
            verbose = TRUE,
            accelerator = accelerator(cpu = TRUE)
          )
      }


      if (save.model == TRUE) {
        luz_save(
          BinaryModel,
          paste(
            output.data.path,
            trainingfolder,
            n.epoch,
            architecture,
            "model.pt",
            sep = "_"
          )
        )
      }


      TempCSV.TrainedModel <-
        read.csv(
          paste(
            output.data.path,
            trainingfolder,
            n.epoch,
            architecture,
            "logs_model.csv",
            sep = "_"
          )
        )
      TrainedModel.loss <-
        TempCSV.TrainedModel[nrow(TempCSV.TrainedModel), ]$loss

      LossPlot <-
        ggline(
          data = TempCSV.TrainedModel,
          x = "epoch",
          y = "loss",
          color = "set"
        )
      print(LossPlot)

      # Calculate performance metrics for TrainedModel -------------------------------------
      dir.create(paste(output.data.path, "performance_tables", sep = ""),
        showWarnings = FALSE
      )

      # Get the list of image files
      imageFiles <-
        list.files(paste(test.data, sep = ""),
          recursive = TRUE,
          full.names = TRUE
        )
      imageFileShort <-
        list.files(paste(test.data, sep = ""),
          recursive = TRUE,
          full.names = FALSE
        )
      Folder <- str_split_fixed(imageFileShort, pattern = "/", n = 2)[, 1]
      imageFileShort <-
        str_split_fixed(imageFileShort, pattern = "/", n = 2)[, 2]

      # Prepare output tables
      outputTableTrainedModel <- data.frame()


      # Define transforms based on model type
      if (str_detect(architecture, pattern = "resnet")) {
        transform_list <- . %>%
          torchvision::transform_to_tensor() %>%
          torchvision::transform_color_jitter() %>%
          transform_resize(256) %>%
          transform_center_crop(224) %>%
          transform_normalize(
            mean = c(0.485, 0.456, 0.406),
            std = c(0.229, 0.224, 0.225)
          )
      } else {
        transform_list <- . %>%
          torchvision::transform_to_tensor() %>%
          torchvision::transform_resize(size = c(224, 224)) %>%
          torchvision::transform_normalize(
            mean = c(0.485, 0.456, 0.406),
            std = c(0.229, 0.224, 0.225)
          )
      }

      test_ds <-
        image_folder_dataset(test.data, transform = transform_list)
      test_dl <-
        dataloader(test_ds, batch_size = batch_size, shuffle = FALSE)

      # Predict using TrainedModel
      TrainedModelPred <- predict(BinaryModel, test_dl)
      TrainedModelProb <- torch_sigmoid(TrainedModelPred)
      TrainedModelProb <-
        as_array(torch_tensor(TrainedModelProb, device = "cpu"))
      TrainedModelClass <-
        ifelse((TrainedModelProb) < 0.5, positive.class, negative.class)

      # Add the results to output tables
      outputTableTrainedModel <-
        rbind(
          outputTableTrainedModel,
          data.frame(
            Label = Folder,
            Probability = TrainedModelProb,
            PredictedClass = TrainedModelClass,
            ActualClass = Folder
          )
        )

      outputTableTrainedModel$Probability <-
        1 - outputTableTrainedModel$Probability
      # Save the output table as CSV file
      write.csv(
        outputTableTrainedModel,
        paste(
          output.data.path,
          trainingfolder,
          n.epoch,
          "output_TrainedModel.csv",
          sep = "_"
        ),
        row.names = FALSE
      )

      message(
        paste(
          "Here are actual class labels, if they do not contain the positive or negative class cannot evaluate model performance:",
          unique(outputTableTrainedModel$ActualClass)
        )
      )

      # Initialize data frames
      CombinedTempRow <- data.frame()
      TransferLearningCNNDF <- data.frame()
      thresholds <- list.thresholds

      binarylabels <- ifelse(outputTableTrainedModel$ActualClass == positive.class, 1, 0)

      for (threshold in thresholds) {
        # TrainedModel
        TrainedModelPredictedClass <-
          ifelse((outputTableTrainedModel$Probability) >= threshold,
            positive.class,
            negative.class
          )
        TrainedModelPredictedClass <-
          factor(TrainedModelPredictedClass, levels = levels(as.factor(
            outputTableTrainedModel$ActualClass
          )))

        TrainedModelPerf <- caret::confusionMatrix(
          as.factor(TrainedModelPredictedClass),
          as.factor(outputTableTrainedModel$ActualClass),
          mode = "everything"
        )$byClass

        TempRowTrainedModel <- cbind.data.frame(
          t(TrainedModelPerf),
          TrainedModel.loss,
          trainingfolder,
          n.epoch,
          architecture
        )

        colnames(TempRowTrainedModel) <- c(
          "Sensitivity",
          "Specificity",
          "Pos Pred Value",
          "Neg Pred Value",
          "Precision",
          "Recall",
          "F1",
          "Prevalence",
          "Detection Rate",
          "Detection Prevalence",
          "Balanced Accuracy",
          "Validation loss",
          "Training Data",
          "N epochs",
          "CNN Architecture"
        )

        TempRowTrainedModel$Threshold <- as.character(threshold)
        CombinedTempRow <-
          rbind.data.frame(CombinedTempRow, TempRowTrainedModel)
      }


      ROCRpred <- ROCR::prediction(predictions = as.numeric(outputTableTrainedModel$Probability), labels = binarylabels)
      AUCval <- ROCR::performance(ROCRpred, "auc")

      CombinedTempRow$AUC <- as.numeric(AUCval@y.values)

      TransferLearningCNNDF <-
        rbind.data.frame(TransferLearningCNNDF, CombinedTempRow)
      TransferLearningCNNDF$Frozen <- unfreeze.param
      TransferLearningCNNDF$Class <- positive.class
      TransferLearningCNNDF$TestDataPath <- test.data[1]
      filename <-
        paste(
          output.data.path,
          "performance_tables/",
          trainingfolder,
          "_",
          n.epoch,
          "_",
          architecture,
          "_CNNDF.csv",
          sep = ""
        )
      write.csv(TransferLearningCNNDF, filename, row.names = FALSE)

      rm(BinaryModel)
    }
  }
