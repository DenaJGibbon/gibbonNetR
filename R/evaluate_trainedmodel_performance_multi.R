#' Evaluate Model Performance on Image Data
#'
#' Given trained models and a set of images, this function evaluates the performance of the models.
#'
#' @param trained_models_dir Path to the directory containing trained models (.pt files).
#' @param image_data_dir Path to the directory containing image data for evaluation.
#' @param output_dir Path to the directory where the performance scores will be saved.
#' @param class_names Character vector specifying class names. User specified from training data folders.
#' @param noise.category Category label for noise class. Default is 'noise'.
#' @param unfreeze Logical indicating whether to unfreeze model parameters. User specified based on trained model.
#'
#' @return The .csv files containing summary of performance are written to output_dir.
#' @importFrom stringr str_split_fixed str_detect
#' @examples {
#'   # Set directory paths for trained models and test images
#'   trained_models_dir <- system.file("extdata", "trainedresnetmulti", package = "gibbonNetR")
#'   image_data_dir <- system.file("extdata", "multiclass", "test", package = "gibbonNetR")
#'   class_names <- c("female.gibbon", "hornbill.helmeted", "hornbill.rhino", "long.argus", "noise")
#'
#'   # Evaluate the performance of the trained models using the test images
#'   evaluate_trainedmodel_performance_multi(
#'     trained_models_dir = trained_models_dir,
#'     class_names = class_names,
#'     image_data_dir = image_data_dir,
#'     output_dir = file.path(tempdir(), "data/"),
#'     noise.category = "noise"
#'   )
#'
#'   # Find the location of saved evaluation files
#'   CSVName <- list.files(file.path(tempdir(), "data"), recursive = TRUE, full.names = TRUE)
#'
#'   # Check the output of the first file
#'   head(read.csv(CSVName[1]))
#' }
#'
#' @importFrom purrr %>%
#' @importFrom utils write.csv read.csv
#' @import data.table
#' @importFrom tidyr pivot_longer
#' @note Takes the directory of models trained 'train_CNN_multi'
#' and test folder created using 'spectrogram_images'.
#' @export
#'
evaluate_trainedmodel_performance_multi <-
  function(trained_models_dir,
           image_data_dir,
           output_dir = "data/",
           class_names,
           noise.category = "noise",
           unfreeze = TRUE) {
    # List trained models
    trained_models <-
      list.files(
        trained_models_dir,
        pattern = ".pt",
        full.names = TRUE,
        recursive = T
      )

    if (length(trained_models) == 0) {
      message("No models in specified directory")
      stop("Stopping execution: No models found.") # Stops the execution
    }


    # List image files
    image_files <-
      list.files(image_data_dir,
        recursive = TRUE,
        full.names = TRUE
      )
    image_files_short <-
      list.files(image_data_dir,
        recursive = TRUE,
        full.names = FALSE
      )

    # Loop through each trained model
    for (model_path in trained_models) {
      performance_scores <- data.frame()
      model <- luz_load(model_path)

      model_name <- basename(model_path)
      training_data <-
        str_split_fixed(model_name, pattern = "_", n = 4)[, 2]
      n_epochs <-
        str_split_fixed(model_name, pattern = "_", n = 4)[, 3]
      model_type <-
        str_split_fixed(
          str_split_fixed(model_name, pattern = "_", n = 4)[, 4],
          pattern = ".pt",
          n = 2
        )[, 1]

      message(paste(
        "Evaluating performance of",
        model_type,
        "N epochs=",
        n_epochs
      ))
      # Evaluate model on each image file

      Folder <- sapply(image_files_short, function(x) {
        dirname(x)
      })

      # Define transforms based on model type
      if (str_detect(model_type, pattern = "ResNet")) {
        transform_list <- . %>%
          torchvision::transform_to_tensor() %>%
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
        image_folder_dataset(image_data_dir, transform = transform_list)
      test_dl <- dataloader(test_ds, batch_size = 32, shuffle = FALSE)

      # Predict using TrainedModel
      TrainedModelPred <- predict(model, test_dl)

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


      UniqueClasses <- class_names
      UniqueClasses <- UniqueClasses[-which(UniqueClasses == noise.category)]

      # Initialize data frames
      CombinedTempRow <- data.frame()
      TransferLearningCNNDF <- data.frame()
      thresholds <- seq(0.1, 1, 0.1)

      for (b in 1:length(UniqueClasses)) {
        outputTableMultiSub <- outputTableMulti

        # Convert to long format
        prob_long <- outputTableMultiSub %>%
          pivot_longer(
            cols = -ActualClass,  # Convert all columns except ActualClass
            names_to = "PredictedClass",
            values_to = "Probability"
          )

        prob_long$Probability <-
          ifelse(prob_long$PredictedClass == "Noise", 1 -
                   prob_long$Probability, prob_long$Probability)

        prob_long <-
          subset(prob_long,ActualClass== UniqueClasses[b] |ActualClass== noise.category )

        binary_labels <- ifelse(prob_long$ActualClass == UniqueClasses[b], 1, 0)

        if(sum(binary_labels)==0){
          message(paste('Skipping',UniqueClasses[b], 'cannot calcuate performance'))
        } else {
          pred <- prediction(prob_long$Probability , binary_labels)
          AUCval <- performance(pred, measure = "auc")

          for (threshold in thresholds) {
            MultiPredictedClass <- ifelse((prob_long$Probability > threshold), UniqueClasses[b], noise.category)

            MultiPredictedClass <- factor(MultiPredictedClass, levels = levels(as.factor(prob_long$ActualClass)))

            MultiPerf <- caret::confusionMatrix(
              as.factor(MultiPredictedClass),
              as.factor(prob_long$ActualClass),
              mode = "everything"
            )$byClass

            TempRowMulti <- cbind.data.frame(
              t(MultiPerf),
              training_data,
              n_epochs,
              model_type
            )

            colnames(TempRowMulti) <- c(
              "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
              "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
              "Detection Prevalence", "Balanced Accuracy",
              "Training Data",
              "N epochs",
              "CNN Architecture"
            )

            TempRowMulti$AUC <- as.numeric(AUCval@y.values)
            TempRowMulti$Threshold <- as.character(threshold)
            TempRowMulti$Class <- UniqueClasses[b]
            TempRowMulti$Class <- as.factor(TempRowMulti$Class)
            TempRowMulti$TestDataPath <- image_data_dir
            CombinedTempRow <- rbind.data.frame(CombinedTempRow, TempRowMulti)
          }

        }
      }


      filename <-
        paste(
          output_dir,
          "performance_tables_multi_trained/",
          training_data,
          "_",
          n_epochs,
          "_",
          model_type,
          "_TransferLearningTrainedModel.csv",
          sep = ""
        )

      filename_multi <-
        paste(
          output_dir,
          "/performance_tables_multi_trained_combined/",
          training_data,
          "_",
          n_epochs,
          "_",
          model_type,
          "_TransferLearningCNNDFmulti.csv",
          sep = ""
        )

      dir.create(
        paste(output_dir, "/performance_tables_multi_trained/", sep = ""),
        showWarnings = FALSE,
        recursive = T
      )

      # Return the index of the max values (i.e. which class)
      PredictTop1 <- torch_argmax(TrainedModelPred, dim = 2)

      # Save to cpu
      PredictTop1 <-
        as_array(torch_tensor(PredictTop1, device = "cpu"))

      # Convert to a factor
      PredictTop1 <- as.factor(PredictTop1)

      PredictTop1Names <-
        droplevels(factor(
          PredictTop1,
          levels = 1:length(class_names),
          labels = class_names
        ))
      Folder <-
        factor(Folder, levels = c(levels(Folder), class_names))

      PredictTop1Names <-
        factor(PredictTop1Names, levels = levels(as.factor(Folder)))

      # Create confusion matrix with filtered predictions
      ConfMatrix <- caret::confusionMatrix(
        data = PredictTop1Names,
        reference = as.factor(Folder)
      )


      CombinedTempRow$Top1Accuracy <-
        as.numeric(ConfMatrix$overall[1])

      write.csv(CombinedTempRow, filename, row.names = FALSE)

      rm(model)
    }
  }
