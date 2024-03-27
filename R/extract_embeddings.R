#' Extract Embeddings and Create Scatter Plots
#'
#' This function loads a fine-tuned PyTorch model, extracts embeddings from a set of test images,
#' performs dimensionality reduction using UMAP, and creates scatter plots to visualize the embeddings.
#'
#' @param test_input A character string specifying the path to the directory containing the test images.
#' @param model_path A character string specifying the path to the pre-trained PyTorch model file.
#' @param target_class A character string specifying the class of interest for cluster analysis.
#'
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
#' # Example usage:
#' result <- extract_embeddings("data/imagesmalaysiamulti/test",
#'   "/Users/denaclink/Desktop/RStudioProjects/Gibbon-transfer-learning-multispecies/model_output/_imagesmulti_multi_unfrozen_TRUE_/_imagesmulti_5_resnet18_model.pt",
#'   target_class = "duet"
#' )
#' print(result)
#'}
#' @export
# Define the function
extract_embeddings <- function(test_input, model_path, target_class) {
  # Load the fine-tuned model
  fine_tuned_model <- luz_load(model_path)


  # Create a dataset from the test images
  test_ds <- image_folder_dataset(
    file.path(test_input),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(size = c(224, 224)) %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
    target_transform = function(x) as.double(x) - 1
  )

  # Create a dataloader
  test_dl <- dataloader(test_ds, batch_size = 32, shuffle = FALSE)

  # Define the module
  net <- torch::nn_module(
    initialize = function() {
      self$model <- fine_tuned_model
      self$feature_extractor <- nn_sequential(
        self$model$features,
        self$model$avgpool,
        nn_flatten(start_dim = 2),
        self$model$classifier[1:6],
        nn_linear(150528, 1024)
      )
      for (par in self$parameters) {
        par$requires_grad_(FALSE)
      }
    },
    forward = function(x) {
      x %>% self$feature_extractor()
    }
  )

  # Create a new instance of the module
  net <- net()

  print("processing embeddings")
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

  EmbeddingsM2.umap <- umap::umap(Embeddings[, -c(1025)], controlscale = TRUE, scale = 3, n_neighbors = 5)

  plot.for.EmbeddingsM2 <- cbind.data.frame(EmbeddingsM2.umap$layout[, 1:2], Embeddings$Label)
  colnames(plot.for.EmbeddingsM2) <- c("Dim.1", "Dim.2", "Class")

  EmbeddingsM2Scatter <- ggpubr::ggscatter(data = plot.for.EmbeddingsM2, x = "Dim.1", y = "Dim.2", color = "Class") +
    scale_color_manual(values = viridis::viridis(length(unique(plot.for.EmbeddingsM2$Class)))) +
    ggtitle(paste("True labels")) +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    ) +
    theme(plot.title = element_text(hjust = 1)) +
    theme(plot.title = element_text(hjust = 1))

  TempCluster <- hdbscan(EmbeddingsM2.umap$layout[, 1:2], minPts = 15)

  plot.for.EmbeddingsM2$cluster <- as.factor(TempCluster$cluster)

  EmbeddingsM2ScatterUnsuper <- ggpubr::ggscatter(data = plot.for.EmbeddingsM2, x = "Dim.1", y = "Dim.2", color = "cluster") +
    scale_color_manual(values = viridis::viridis(length(unique(plot.for.EmbeddingsM2$cluster)))) +
    ggtitle(paste("Unsupervised clustering")) +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    ) +
    theme(plot.title = element_text(hjust = 1)) +
    theme(plot.title = element_text(hjust = 1))

  EmbeddingsCombined <- cowplot::plot_grid(EmbeddingsM2Scatter, EmbeddingsM2ScatterUnsuper)

  # Find the cluster with the most observations of the target class
  class_counts <- table(Embeddings$Label, TempCluster$cluster)

  if (target_class %in% Embeddings$Label == FALSE) {
    print("target_class not included in test_input folder names")
    break
  }

  cluster_with_most_class <- colnames(class_counts)[which.max(class_counts[target_class, ])]

  Binary <- ifelse(TempCluster$cluster == cluster_with_most_class, target_class, "Noise")
  BinaryLabels <- ifelse(Embeddings$Label == target_class, target_class, "Noise")


  ConfMat <- caret::confusionMatrix(
    as.factor(Binary),
    as.factor(BinaryLabels),
    mode = "everything"
  )

  print(paste("Unupervised clustering for", target_class))
  print(ConfMat$byClass)

  return(list(
    EmbeddingsCombined = EmbeddingsCombined,
    NMI = NMI(Binary, BinaryLabels),
    ConfusionMatrixUnsupervisedAssigment = ConfMat$byClass
  ))
}
