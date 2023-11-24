#' Train multi CNN Models
#'
#' This function is designed to train CNN models (alexnet, VGG16, VGG19, ResNet18, ResNet50, or ResNet152) on a given dataset.
#' The model is saved, and other metadata is stored for further usage.
#'
#' @param input.data.path Character. The path to the input data folder.
#' @param test.data Character. The path to the test data folder.
#' @param architecture Character. The CNN architecture to use ('alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152').
#' @param unfreeze Logical. Determines if all layers of the pretrained CNN should be unfrozen for retraining.
#'                 Default is TRUE.
#' @param learning_rate Numeric. The learning rate for training the model.
#' @param batch_size Numeric. Batch size for training model.
#' @param epoch.iterations List of integers. The number of epochs for training the model. Default is 1.
#' @param early.stop Character. Determines whether early stopping should be applied or not.
#'                   "yes" to apply and "no" to skip. Default is 'yes'.
#' @param output.base.path Character. The base path where the output files should be saved.
#'                          Default is 'data/'.
#' @param save.model Logical. Whether to save model for future use.
#' @param trainingfolder Character. A shortened descriptor of the training data, used for naming output files.
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
#'   train_CNN_multi(
#'     input.data.path = "path_to_input_data",
#'     test.data = "path_to_test_data",
#'     architecture = "alexnet",  # Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'
#'     unfreeze = TRUE,
#'     learning_rate = 0.001,
#'     epoch.iterations = 1,  # Or any other list of integer epochs
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

train_CNN_multi <- function(input.data.path, test.data, architecture,
                             unfreeze = TRUE, batch_size=32, learning_rate,
                             save.model= FALSE,
                             epoch.iterations=1, early.stop = 'yes',
                             output.base.path = 'data/',
                             trainingfolder,
                             noise.category = "Noise") {

  # Device
  device <- if(cuda_is_available()) "cuda" else "cpu"

  to_device <- function(x, device) {
    x$to(device = device)
  }

  # Location to save the output
  output.data.path <- paste(output.base.path, trainingfolder, 'multi', 'unfrozen', unfreeze,  '/', sep='_')

  # Create if doesn't exist
  dir.create(output.data.path, showWarnings = FALSE)

  # Metadata
  metadata <- tibble(
    Model_Name = architecture,
    Training_Data_Path = input.data.path,
    Test_Data_Path = test.data,
    Output_Path = output.data.path,
    Device_Used = device,
    EarlyStop = early.stop,
    Layers_Unfrozen = unfreeze,
    Epochs = epoch.iterations,
    Learning_rate=learning_rate,
    Noise.class=noise.category
  )

  write_csv(metadata, paste0(output.data.path, architecture, "_model_metadata.csv"))

  for(a in 1:length(epoch.iterations )){
    print(paste('Training', architecture))
    n.epoch <- epoch.iterations [a]


    if (architecture %in% c("alexnet", "vgg16", "vgg19")==TRUE) {
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

    }

    if (architecture %in% c("resnet18", "resnet50", "resnet152")==TRUE) {
      train_ds <- image_folder_dataset(
        file.path(input.data.path,'train' ),
        transform = . %>%
          torchvision::transform_to_tensor() %>%
          torchvision::transform_color_jitter() %>%
          transform_resize(256) %>%
          transform_center_crop(224) %>%
          transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)))

      valid_ds <- image_folder_dataset(
        file.path(input.data.path, "valid"),
        transform = . %>%
          torchvision::transform_to_tensor() %>%
          transform_resize(256) %>%
          transform_center_crop(224) %>%
          transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)))

    }


    train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE, drop_last = TRUE)
    valid_dl <- dataloader(valid_ds, batch_size = batch_size, shuffle = FALSE, drop_last = TRUE)

    # Read class names
    class_names <- sort(unique(attr(train_ds$class_to_idx, "names")))
    num_classes <- length(class_names)
    cat('Detected classes:', paste(class_names, collapse = ', '), '\n')


    # Model setup
    if (architecture == "alexnet") {
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
            nn_linear(256, num_classes)
          )
        },
        forward = function(x) {
          output <- self$model(x)
          torch_squeeze(output, dim=2)
        }
      )

    } else if (architecture == "vgg16") {
      net <- torch::nn_module(
        initialize = function() {
          self$model <- model_vgg16 (pretrained = TRUE)

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
          torch_squeeze(output, dim=2)
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
          torch_squeeze(output, dim=2)
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
            nn_linear(1024, num_classes)
          )
        },
        forward = function(x) {
          output <- self$model(x)
          torch_squeeze(output, dim=2)
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
            nn_linear(1024, num_classes)
          )
        },
        forward = function(x) {
          output <- self$model(x)
          torch_squeeze(output, dim=2)
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
            nn_linear(1024, num_classes)
          )
        },
        forward = function(x) {
          output <- self$model(x)
          torch_squeeze(output, dim=2)
        }
      )
    } else {
      stop("Invalid architecture specified. Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'.")
    }


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
      multiModel <- fitted %>%
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
              luz_callback_csv_logger(paste(output.data.path, trainingfolder, n.epoch, architecture, "logs_model.csv", sep = '_'))
            ),
            verbose = TRUE
        )
    } else {
      multiModel <- fitted %>%
        fit(train_dl, epochs = n.epoch, valid_data = valid_dl,
            callbacks = list(
              luz_callback_lr_scheduler(
                lr_one_cycle,
                max_lr =learning_rate,
                epochs = n.epoch,
                steps_per_epoch = length(train_dl),
                call_on = "on_batch_end"
              ),
              luz_callback_csv_logger(paste(output.data.path, trainingfolder, n.epoch, architecture, "logs_model.csv", sep = '_'))
            ),
            verbose = TRUE
        )
    }


    if(save.model==TRUE){
      luz_save(multiModel, paste( output.data.path,trainingfolder,n.epoch,architecture, "model.pt",sep='_'))
    }


    TempCSV.TrainedModel <- read.csv(paste(output.data.path, trainingfolder, n.epoch, architecture, "logs_model.csv", sep = '_'))
    TrainedModel.loss <- TempCSV.TrainedModel[nrow(TempCSV.TrainedModel),]$loss


    LossPlot <- ggline(data=TempCSV.TrainedModel,x='epoch',y='loss',color = 'set')
    print(LossPlot)

    # Get the list of image files
    imageFiles <- list.files(paste(test.data,sep=''), recursive = TRUE, full.names = TRUE)
    imageFileShort <- list.files(paste(test.data,sep=''), recursive = TRUE, full.names = FALSE)
    Folder <- str_split_fixed(imageFileShort,pattern = '/',n=2)[,1]
    imageFileShort <- str_split_fixed(imageFileShort,pattern = '/',n=2)[,2]

    # Prepare output tables
    outputTableMultiSub <- data.frame()

    # Define transforms based on model type
    if (str_detect(architecture, pattern = 'resnet')) {
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

    test_ds <- image_folder_dataset(test.data, transform = transform_list)
    test_dl <- dataloader(test_ds, batch_size = batch_size, shuffle =FALSE)

    # Predict using TrainedModel
    TrainedModelPred <- predict(multiModel, test_dl)

    # Return the index of the max values (i.e. which class)
    PredMPS <- torch_argmax(TrainedModelPred, dim = 2)

    # Save to cpu
    PredMPS <- as_array(torch_tensor(PredMPS, device = 'cpu'))

    # Convert to a factor
    modelMultiPred <- as.factor(PredMPS)
    print(modelMultiPred)

    # Calculate the probability associated with each class
    Probability <- as_array(torch_tensor(nnf_softmax(TrainedModelPred, dim = 2), device = 'cpu'))

    # Find the index of the maximum value in each row
    max_prob_idx <- apply(Probability, 1, which.max)

    # Map the index to actual probability
    predicted_class_probability <- sapply(1:nrow(Probability), function(i) Probability[i, max_prob_idx[i]])

    # Convert the integer predictions to factor and then to character based on the levels
    modelMultiNames <- factor(modelMultiPred, levels = 1:length(class_names), labels = class_names)

    outputTableMulti <- cbind.data.frame(modelMultiNames, predicted_class_probability)
    colnames(outputTableMulti) <- c('PredictedClass', 'Probability')
    outputTableMulti$ActualClass <- Folder

    # Save the output table as CSV file
    write.csv(outputTableMulti, paste(output.data.path,trainingfolder,n.epoch,architecture, "output_Multi.csv", sep = '_'), row.names = FALSE)

    UniqueClasses <- unique(outputTableMulti$ActualClass)
    Probability <- as.data.frame(Probability)
    colnames(Probability) <- UniqueClasses
    UniqueClasses <- UniqueClasses[-which(UniqueClasses == noise.category)]

    # Initialize data frames
    CombinedTempRow <- data.frame()
    TransferLearningCNNDF <- data.frame()
    thresholds <- seq(0.1, 1, 0.1)

    for (b in 1:length(UniqueClasses)) {


      outputTableMultiSub <-outputTableMulti
      outputTableMultiSub$Probability <- Probability[,c(UniqueClasses[b] )]

      outputTableMultiSub$ActualClass <-
        ifelse(outputTableMultiSub$ActualClass==UniqueClasses[b],UniqueClasses[b],noise.category)


      for (threshold in thresholds) {
        MultiPredictedClass <- ifelse((outputTableMultiSub$Probability > threshold ), UniqueClasses[b], noise.category)


        MultiPerf <- caret::confusionMatrix(
          as.factor(MultiPredictedClass),
          as.factor(outputTableMultiSub$ActualClass),
          mode = 'everything'
        )$byClass

        TempRowMulti <- cbind.data.frame(
          t(MultiPerf),
           TrainedModel.loss ,
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

        ROCCurve <- roc(outputTableMultiSub$ActualClass, outputTableMultiSub$Probability)
        TempRowMulti$AUC <-  as.numeric(ROCCurve$auc )
        TempRowMulti$Threshold <- as.character(threshold)
        TempRowMulti$Frozen <- unfreeze
        TempRowMulti$Class <- UniqueClasses[b]
        TempRowMulti$Class <- as.factor(TempRowMulti$Class)
        CombinedTempRow <- rbind.data.frame(CombinedTempRow, TempRowMulti)
      }
    }

    # Location to save the output
    output.data.performance <- paste(output.data.path,'performance_tables_multi/',sep='')

    print(output.data.performance)
    # Create if doesn't exist
    dir.create(output.data.performance, showWarnings = FALSE,recursive = T)

    filename <- paste(output.data.performance, trainingfolder, '_', n.epoch, '_', architecture, '_TransferLearningCNNDFMultiThreshold.csv', sep = '')
    write.csv(CombinedTempRow, filename, row.names = FALSE)

    rm(multiModel)
  }
}

