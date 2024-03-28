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
#' @export
evaluate_trainedmodel_performance_multi <- function(trained_models_dir, image_data_dir, output_dir='data/',trainingfolder,
                                                    batch_size=32,
                                                    class_names = c('duet','hornbill.helmeted','hornbill.rhino','long.argus','noise'),
                                                    noise.category='noise',unfreeze='TRUE') {

  # List trained models
  trained_models <- list.files(trained_models_dir, pattern = '.pt', full.names = TRUE,recursive = T)

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

    print(paste('Evaluating performance of', model_type, 'N epochs=',n_epochs))
    # Evaluate model on each image file

    Folder <- sapply(image_files_short, function(x) dirname(x))

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

    test_ds <- image_folder_dataset(image_data_dir, transform = transform_list)
    test_dl <- dataloader(test_ds, batch_size = batch_size, shuffle =FALSE)


    # Predict using trained model
    Pred <- predict(model, test_dl)

    # Calculate the probability associated with each class
    Probability <- as_array(torch_tensor(nnf_softmax(Pred, dim = 2), device = 'cpu'))

    Probability <- as.data.frame(Probability)
    colnames(Probability) <- class_names

    Probability$ActualClass <- Folder

    UniqueClasses <- unique(Probability$ActualClass)

    UniqueClasses <- UniqueClasses[-which(UniqueClasses == noise.category)]

    # Initialize data frames
    CombinedTempRow <- data.frame()

    for (b in 1:length(UniqueClasses)) {
      print(UniqueClasses[b])
      outputTableSub <-Probability[,c(UniqueClasses[b],'ActualClass' )]
      outputTableSub$Probability <- outputTableSub[,1]

      outputTableSub$ActualClass <-
        ifelse(outputTableSub$ActualClass==UniqueClasses[b],UniqueClasses[b],noise.category)

        ROCRpred <- ROCR::prediction(predictions = outputTableSub$Probability, labels = as.factor(outputTableSub$ActualClass),
                                     label.ordering =c(UniqueClasses[b],noise.category) )

        AUCval <- ROCR::performance(ROCRpred,'aucpr')
        F1val <- ROCR::performance(ROCRpred,'f')
        F1 <- F1val@y.values[[1]]
        Rec <- ROCR::performance(ROCRpred, "rec")
        Recall <-Rec@y.values[[1]]
        Prec <- ROCR::performance(ROCRpred, "prec")
        Precision <- Prec@y.values[[1]]

        perf <- ROCR::performance(ROCRpred,"fpr")
        FPR <- perf@y.values[[1]]

        AUC <- AUCval@y.values[[1]]

        Threshold <- ROCRpred@cutoffs[[1]] # unique prediction scores (sorted in descending order) at which the true positive and false positive counts change

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

        TempRow$Class <- UniqueClasses[b]
        TempRow$TestData <- str_replace_all(image_data_dir,pattern = '/',replacement ='_')

        CombinedTempRow <- rbind.data.frame(CombinedTempRow, TempRow)
      }

    # Return the index of the max values (i.e. which class)
    PredictTop1 <- torch_argmax(Pred, dim = 2)

    # Save to cpu
    PredictTop1 <- as_array(torch_tensor(PredictTop1, device = 'cpu'))

    # Convert to a factor
    PredictTop1<- as.factor(PredictTop1)
    print(PredictTop1)

    PredictTop1Names <- droplevels(factor(PredictTop1, levels = 1:length(class_names), labels = class_names))

    ConfMatrix <- caret::confusionMatrix(data=PredictTop1Names,
                           reference=as.factor(as.character(Folder)))

    CombinedTempRow$Top1Accuracy <-  as.numeric(ConfMatrix$overall[1])

    filename <- paste(output_dir,'performance_tables_multi_trained/', training_data, '_', n_epochs, '_', model_type, '_TransferLearningTrainedModel.csv', sep = '')

      dir.create(paste(output_dir, '/performance_tables_multi_trained/', sep = ''),recursive = T)

      write.csv(CombinedTempRow, filename, row.names = FALSE)

      rm(model)
    }
  }

