% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/train_CNN_binary.R
\name{train_CNN_binary}
\alias{train_CNN_binary}
\title{Train Binary CNN Models}
\usage{
train_CNN_binary(
  input.data.path,
  test.data,
  architecture,
  noise.weight = 0.5,
  brightness = 0,
  contrast = 0,
  saturation = 0,
  unfreeze.param = TRUE,
  batch_size = 32,
  learning_rate,
  save.model = FALSE,
  epoch.iterations = 1,
  early.stop = "yes",
  output.base.path = "data/",
  trainingfolder,
  list.thresholds = seq(0.1, 1, 0.1),
  positive.class = "Gibbons",
  negative.class = "Noise"
)
}
\arguments{
\item{input.data.path}{Character. The path to the folder containing the training data.}

\item{test.data}{Character. The path to the folder containing the test data.}

\item{architecture}{Character. The CNN architecture to use ('alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152').}

\item{noise.weight}{Numeric. Assigned weight for the noise class. Default is 0.5.}

\item{brightness}{Numeric. Brightness adjustment factor for color jitter. A value of 0 means no change. Higher values increase brightness. Default is 0.}

\item{contrast}{Numeric. Contrast adjustment factor for color jitter. A value of 0 means no change. Higher values increase contrast. Default is 0.}

\item{saturation}{Numeric. Saturation adjustment factor for color jitter. A value of 0 means no change. Higher values increase color saturation. Default is 0.}

\item{unfreeze.param}{Logical. Determines whether to unfreeze all layers of the pretrained CNN for retraining. Default is TRUE.}

\item{batch_size}{Numeric. Batch size for training the model. Default is 32.}

\item{learning_rate}{Numeric. The learning rate for training the model.}

\item{save.model}{Logical. Whether to save the trained model for future use. Default is FALSE.}

\item{epoch.iterations}{Numeric. The number of epochs for training the model. Default is 1.}

\item{early.stop}{Character. Determines whether early stopping should be applied or not. Options: "yes" or "no". Default is 'yes'.}

\item{output.base.path}{Character. The base path where the output files should be saved. Default is 'data/'.}

\item{trainingfolder}{Character. A descriptor of the training data used for naming output files.}

\item{list.thresholds}{Numerical list indicating thresholds. Default is seq(0.1,1,.1).}

\item{positive.class}{Character. The name of the positive class label. Default is 'Gibbons'.}

\item{negative.class}{Character. The name of the negative class label. Default is 'Noise'.}
}
\value{
The function generates multiple output files, including:
\itemize{
\item \strong{Trained Models}: If save.model = TRUE outputs saved model files (.pt) for specified architectures.
\item \strong{Training Logs}: logs_model.csv containing logs of training sessions, including loss and accuracy metrics.
\item \strong{Metadata}: model_metadata.csv contains metadata from training run
\item \strong{Model predictions}: Saved for each architecture in output_TrainedModel_testdata.csv
\item \strong{Performance Evaluation}: Saves .csv summarizing performance for each architecture in nested folder in output.base.path 'performance_tables'.
}
}
\description{
This function trains Convolutional Neural Network (CNN) models, such as AlexNet, VGG16, VGG19, ResNet18, ResNet50, or ResNet152, on a given dataset. The trained model is saved along with metadata for further usage.
}
\note{
Requires train, valid, and test folders
created using created using 'spectrogram_images'
}
\examples{
{
  input.data.path <- system.file("extdata", "binary/", package = "gibbonNetR")
  test.data <- system.file("extdata", "binary/test/", package = "gibbonNetR")
  result <- train_CNN_binary(
    input.data.path = input.data.path,
    test.data = test.data,
    architecture = "alexnet", # Choose architecture
    unfreeze.param = TRUE,
    batch_size = 6,
    learning_rate = 0.001,
    epoch.iterations = 1, # Or any other list of integer epochs
    early.stop = "yes",
    output.base.path = paste(tempdir(), "/", sep = ""),
    trainingfolder = "test_binary"
  )
  print(result)
}

}
\seealso{
\code{\link[torch]{nn_module}} and other torch functions.
}
