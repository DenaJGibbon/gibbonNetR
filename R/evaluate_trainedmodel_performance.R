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
#' trained_models_dir <- '/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/inst/extdata/trainedresnetbinary/'
#' image_data_dir <- "inst/extdata/binary/test/"
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

#' @export
evaluate_trainedmodel_performance <- function(trained_models_dir, image_data_dir, output_dir='data/',
                                              positive.class='Gibbons',negative.class='Noise') {

  # List trained models
  trained_models <- list.files(trained_models_dir, pattern = '.pt', full.names = TRUE)

  # List image files
  image_files <- list.files(image_data_dir, recursive = TRUE, full.names = TRUE)
  image_files_short <- list.files(image_data_dir, recursive = TRUE, full.names = FALSE)

  # Loop through each trained model
  for (model_path in trained_models) {
    # Initialize data frames
    CombinedTempRow <- data.frame()
    TransferLearningCNNDF <- data.frame()

    performance_scores <- data.frame()
    model <- luz_load(model_path)

    model_name <- basename(model_path)
    training_data <- str_split_fixed(model_name, pattern = '_', n = 4)[,2]
    n_epochs <- str_split_fixed(model_name, pattern = '_', n = 4)[,3]
    model_type <- str_split_fixed(str_split_fixed(model_name, pattern = '_', n = 4)[,4], pattern = '.pt', n = 2)[,1]

    # Evaluate model on each image file

      actual_labels <- sapply(image_files_short, function(x) dirname(x))

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
      #probs <- 1- probs

      ROCRpred <-  ROCR::prediction(predictions = probs,
                                    labels = actual_labels,
                                    label.ordering =)
      AUCval <- ROCR::performance(ROCRpred,'aucpr')
      F1val <- ROCR::performance(ROCRpred,'f')
      F1 <- F1val@y.values[[1]]
      Rec <- ROCR::performance(ROCRpred, "rec",window.size=50)
      Recall <-Rec@y.values[[1]]
      Prec <- ROCR::performance(ROCRpred, "prec")
      Precision <- Prec@y.values[[1]]

      perf <- ROCR::performance(ROCRpred,"fpr")
      FPR <- perf@y.values[[1]]

      AUC <- AUCval@y.values[[1]]

      Threshold <- ROCRpred@cutoffs[[1]]

      TempRow <- cbind.data.frame(F1,Recall,Precision,FPR,AUC,Threshold,
                                  training_data,
                                   n_epochs,
                                   model_type)

      colnames(TempRow) <- c(

        "F1","Precision", "Recall", "FPR", "AUC","Threshold",
        "Training Data",
        "N epochs",
        "CNN Architecture"
      )

     InfIndex <-  which(is.infinite(TempRow$Threshold))

    if(is.numeric(InfIndex)  ==T){
       TempRow <- TempRow[-InfIndex,]
     }

     TempRow$Class <- positive.class
     TempRow$TestData <- str_replace_all(image_data_dir,pattern = '/',replacement ='_')
      TransferLearningCNNDF <- rbind.data.frame(TransferLearningCNNDF, TempRow)
      filename <- paste(output_dir,'performance_tables_trained/', training_data, '_', n_epochs, '_', model_type, '_TransferLearningTrainedModel.csv', sep = '')
      dir.create(dirname(filename), showWarnings = FALSE,recursive = T)
      write.csv(TransferLearningCNNDF, filename, row.names = FALSE)

  }

  invisible(NULL)
}
