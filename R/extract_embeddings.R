#' Extract Embeddings and Create Scatter Plots
#'
#' This function loads a fine-tuned Torch model, extracts embeddings from a set of test images,
#' performs dimensionality reduction using UMAP, and creates scatter plots to visualize the embeddings.
#'
#' @param test_input A character string specifying the path to the directory containing the test images.
#' @param model_path A character string specifying the path to the pre-trained PyTorch model file.
#' @param target_class A character string specifying the class of interest for cluster analysis.
#' @param unsupervised Logical, indicates whether to assign 'target_class' to a cluster and calculate NMI and corresponding confusion matrix
#' @return A list containing the following components:
#' \describe{
#'   \item{EmbeddingsCombined}{A combined scatter plot of embeddings, showing class and cluster colors.}
#'   \item{NMI}{Normalized Mutual Information (NMI) score between clustering results and ground truth labels.}
#'   \item{ConfusionMatrix}{A confusion matrix showing classification performance metrics.}
#' }
#'
#' @import luz
#' @import torch
#' @import torchvision
#' @import torchdatasets
#' @import stringr
#' @import tuneR
#' @import seewave
#' @import ggplot2
#' @import dbscan
#' @import aricode
#' @import umap
#' @import viridis
#' @import cowplot
#' @import caret
#'
#' @examples {
#'   #' Set model directory
#'   trained_models_dir <- system.file("extdata", "trainedresnetmulti/", package = "gibbonNetR")
#'
#'   #' Specify model path
#'   ModelPath <- list.files(trained_models_dir, full.names = TRUE)
#'
#'   # Specify model path
#'   ImageFile <- system.file("extdata", "multiclass/test/", package = "gibbonNetR")
#'
#'   # Function to extract and plot embeddings
#'   result <- extract_embeddings(
#'     test_input = ImageFile,
#'     model_path = ModelPath,
#'     target_class = "female.gibbon",
#'     unsupervised = "TRUE"
#'   )
#'
#'   print(result$EmbeddingsCombined)
#' }
#' @importFrom utils write.csv read.csv
#' @importFrom coro loop
#' @note Requires a model trained using 'train_CNN_multi' or 'train_CNN_binary',
#' and a directory of spectrogram images created using 'spectrogram_images'.
#' @export

# Define the function
extract_embeddings <- function(test_input,
                               model_path,
                               target_class,
                               unsupervised = "TRUE") {

  # Ensure model_path is a valid file and contains a .pt extension
  if (!file.exists(model_path) || !grepl("\\.pt$", model_path)) {
    stop("Invalid model path: Model file does not exist or is not a .pt file.")
  }

  # Load the fine-tuned model
  fine_tuned_model <- luz_load(model_path)

  # Create a dataset from the test images
  test_ds <- image_folder_dataset(
    file.path(test_input),
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

  # Create a dataloader
  test_dl <- dataloader(test_ds, batch_size = 32, shuffle = FALSE)

  net <- nn_module(
    initialize = function() {
      self$model <- model_resnet18(pretrained = TRUE)

      # Freeze all original ResNet parameters:
      for (par in self$model$parameters) {
        par$requires_grad_(FALSE)
      }

      # Remove the original fully connected layer (fc):
      self$model$fc <- NULL  # Or: self$model$fc <- nn_identity()

      # Create the feature extractor part (up to the avgpool):
      self$feature_extractor <- nn_sequential(
        self$model$conv1,
        self$model$bn1,
        self$model$relu,
        self$model$maxpool,
        self$model$layer1,
        self$model$layer2,
        self$model$layer3,
        self$model$layer4,
        self$model$avgpool,
        nn_flatten(start_dim = 2) # Important: Flatten the output
      )

    },
    forward = function(x) {
      features <- x %>% self$feature_extractor() # Extract features
      return(features) # Return the features directly
    }
  )

  # Create a new instance of the module
  net <- net()

  message("processing embeddings")

  # Extract features
  features <- list()

  coro::loop(for (batch in test_dl) {
    inputs <- batch[[1]]
    with_no_grad({
      outputs <- net$forward(inputs)
    })
    features <- c(features, list(outputs$cpu() %>% as_array()))
  })

  features <- do.call(rbind, features)

  # Read Embeddings.csv
  Embeddings <- as.data.frame(features)

  TempName <- list.files(test_input, recursive = TRUE)

  Embeddings$Label <- dirname(TempName)

  EmbeddingsM2.umap <-
    umap::umap(
      Embeddings[, -c(ncol(Embeddings))],
      controlscale = TRUE,
      scale = 3,
      n_neighbors = 5
    )

  plot.for.EmbeddingsM2 <-
    cbind.data.frame(EmbeddingsM2.umap$layout[, 1:2], Embeddings$Label)

  colnames(plot.for.EmbeddingsM2) <- c("Dim.1", "Dim.2", "Class")

  EmbeddingsM2Scatter <-
    ggpubr::ggscatter(
      data = plot.for.EmbeddingsM2,
      x = "Dim.1",
      y = "Dim.2",
      color = "Class"
    ) +
    scale_color_manual(values = viridis::viridis(length(unique(
      plot.for.EmbeddingsM2$Class
    )))) +
    ggtitle(paste("True labels")) +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    ) +
    theme(plot.title = element_text(hjust = 1)) +
    theme(plot.title = element_text(hjust = 1))

  TempCluster <-
    hdbscan(EmbeddingsM2.umap$layout[, 1:2], minPts = 15)

  plot.for.EmbeddingsM2$cluster <- as.factor(TempCluster$cluster)

  EmbeddingsM2ScatterUnsuper <-
    ggpubr::ggscatter(
      data = plot.for.EmbeddingsM2,
      x = "Dim.1",
      y = "Dim.2",
      color = "cluster"
    ) +
    scale_color_manual(values = viridis::viridis(length(
      unique(plot.for.EmbeddingsM2$cluster)
    ))) +
    ggtitle(paste("Unsupervised clustering")) +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    ) +
    theme(plot.title = element_text(hjust = 1)) +
    theme(plot.title = element_text(hjust = 1)) + guides(color = "none")

  EmbeddingsCombined <-
    cowplot::plot_grid(EmbeddingsM2Scatter, EmbeddingsM2ScatterUnsuper,
      nrow = 2
    )

  if (unsupervised == "TRUE") {
    # Find the cluster with the most observations of the target class
    class_counts <- table(Embeddings$Label, TempCluster$cluster)

    if (target_class %in% Embeddings$Label == FALSE) {
      message("target_class not included in test_input folder names")
      stop("Stopping execution: target_class not found in labels.") # Stops the script
    }


    cluster_with_most_class <-
      colnames(class_counts)[which.max(class_counts[target_class, ])]

    Binary <-
      ifelse(TempCluster$cluster == cluster_with_most_class,
        target_class,
        "Noise"
      )
    BinaryLabels <-
      ifelse(Embeddings$Label == target_class, target_class, "Noise")

    Binary <-
      factor(Binary, levels = levels(as.factor(BinaryLabels)))

    ConfMat <- caret::confusionMatrix(
      as.factor(Binary),
      as.factor(BinaryLabels),
      mode = "everything",
      positive = target_class
    )

    message(paste("Unupervised clustering for", target_class))

    return(
      list(
        EmbeddingsCombined = EmbeddingsCombined,
        NMI = NMI(Binary, BinaryLabels),
        ConfusionMatrixUnsupervisedAssigment = ConfMat$byClass
      )
    )
  } else {
    return(list(EmbeddingsCombined = EmbeddingsCombined))
  }
}
