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
#' @return Invisible NULL. The performance scores are written to the specified output directory.
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
        image_folder_dataset(image_data_dir, transform = transform_list)
      test_dl <- dataloader(test_ds, batch_size = 32, shuffle = FALSE)

      # Predict using trained model
      Pred <- predict(model, test_dl)

      # Calculate the probability associated with each class
      Probability <-
        as_array(torch_tensor(nnf_softmax(Pred, dim = 2), device = "cpu"))

      Probability <- as.data.frame(Probability)
      colnames(Probability) <- class_names

      Probability$ActualClass <- Folder

      UniqueClasses <- unique(Probability$ActualClass)

      UniqueClasses <-
        UniqueClasses[-which(UniqueClasses == noise.category)]

      # Initialize data frames
      CombinedTempRow <- data.frame()

      for (b in 1:length(UniqueClasses)) {
        message(UniqueClasses[b])
        outputTableSub <-
          Probability[, c(UniqueClasses[b], "ActualClass")]

        outputTableSub$Probability <- outputTableSub[, 1]

        outputTableSub$ActualClass <-
          ifelse(outputTableSub$ActualClass == UniqueClasses[b],
            UniqueClasses[b],
            noise.category
          )

        thresholds <- seq(0.1, 1, 0.1)

        for (threshold in thresholds) {
          message(threshold)
          PredictedClass <-
            ifelse((outputTableSub$Probability > threshold),
              UniqueClasses[b],
              noise.category
            )

          PredictedClass <-
            factor(PredictedClass, levels = levels(as.factor(outputTableSub$ActualClass)))

          Perf <- caret::confusionMatrix(
            as.factor(PredictedClass),
            as.factor(outputTableSub$ActualClass),
            mode = "everything",
            positive = UniqueClasses[b]
          )$byClass

          TempRow <- cbind.data.frame(
            t(Perf),
            training_data,
            n_epochs,
            model_type
          )

          colnames(TempRow) <- c(
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
            "Training Data",
            "N epochs",
            "CNN Architecture"
          )

          ROCRpred <- ROCR::prediction(predictions = outputTableSub$Probability, labels = as.factor(outputTableSub$ActualClass))
          AUCval <- ROCR::performance(ROCRpred, "auc")
          TempRow$AUC <- as.numeric(AUCval@y.values)
          TempRow$Threshold <- as.character(threshold)
          TempRow$Frozen <- unfreeze
          TempRow$Class <- UniqueClasses[b]
          TempRow$Class <- as.factor(TempRow$Class)
          CombinedTempRow <-
            rbind.data.frame(CombinedTempRow, TempRow)
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
      PredictTop1 <- torch_argmax(Pred, dim = 2)

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
