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
#' @examples
#' {
#' # Set directory paths for trained models and test images
#'
#' trained_models_dir <- system.file("extdata", "trainedresnetbinary/", package = "gibbonNetR")
#' image_data_dir <- ' system.file("extdata", "binary/test/", package = "gibbonNetR")
#'
#' # Evaluate the performance of the trained models using the test images
#' evaluate_trainedmodel_performance(trained_models_dir = trained_models_dir,
#'                                   image_data_dir = image_data_dir,
#'                                   output_dir = paste(tempdir(), '/data/'),  #' Output directory for evaluation results
#'                                   positive.class = 'Gibbons',  #' Label for positive class
#'                                   negative.class = 'Noise')    #' Label for negative class
#'
#' # Find the location of saved evaluation files
#' CSVName <- list.files(paste(tempdir(), '/data/'), recursive = TRUE, full.names = TRUE)
#'
#' # Check the output of the first file
#' head(read.csv(CSVName[1]))
#'}
#'
#' @export
evaluate_trainedmodel_performance <- function(trained_models_dir, image_data_dir, output_dir='data/',
                                              positive.class='Gibbons',negative.class='Noise') {

  # List trained models
  trained_models <- list.files(trained_models_dir, pattern = '.pt', full.names = TRUE)

  if(length(trained_models)==0){
    print('No models in specified directory')
    break
  }

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

    # Evaluate model on each image file

    actual_labels <- as.character(sapply(image_files_short, function(x) dirname(x)))

    # Define transforms based on model type
    if (str_detect(model_type, pattern = 'resnet')) {
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

    preds <- predict(model, test_dl)

    probs <- as_array(torch_tensor(torch_sigmoid(preds), device = 'cpu'))

    # Switch positive/negative probs
    probs <- 1- probs

    # Initialize data frames
    CombinedTempRow <- data.frame()
    TransferLearningCNNDF <- data.frame()
    thresholds <- seq(0.1,1,0.1)

    for (threshold in thresholds) {
      # TrainedModel
      TrainedModelPredictedClass <- ifelse(probs >= (threshold), positive.class, negative.class)

      TrainedModelPredictedClass <- factor(TrainedModelPredictedClass, levels =levels( as.factor(actual_labels)))

      TrainedModelPerf <- caret::confusionMatrix(
        as.factor(TrainedModelPredictedClass),
        as.factor(actual_labels),
        mode = 'everything',positive=positive.class
      )$byClass

      TempRowTrainedModel <- cbind.data.frame(
        t(TrainedModelPerf),
        'NA',
        training_data,
        n_epochs,
        model_type
      )

      colnames(TempRowTrainedModel) <- c(
        "Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value",
        "Precision", "Recall", "F1", "Prevalence", "Detection Rate",
        "Detection Prevalence", "Balanced Accuracy",
        'Validation Loss',
        "Training Data",
        "N epochs",
        "CNN Architecture"
      )

      TempRowTrainedModel$Class <- positive.class
      TempRowTrainedModel$Threshold <- as.character(threshold)

      CombinedTempRow <- rbind.data.frame(CombinedTempRow, TempRowTrainedModel)
    }

    ROCRpred <-  ROCR::prediction(predictions = probs,
                                  labels = actual_labels)
    AUCval <- ROCR::performance(ROCRpred,'auc')
    CombinedTempRow$AUC <- AUCval@y.values[[1]]

    CombinedTempRow$TestData <- str_replace_all(image_data_dir,pattern = '/',replacement ='_')
    TransferLearningCNNDF <- rbind.data.frame(TransferLearningCNNDF, CombinedTempRow)
    filename <- paste(output_dir,'performance_tables_trained/', training_data, '_', n_epochs, '_', model_type, '_TransferLearningTrainedModel.csv', sep = '')
    dir.create(dirname(filename), showWarnings = FALSE, recursive = T)
    write.csv(TransferLearningCNNDF, filename, row.names = FALSE)

  }

  invisible(NULL)
}
