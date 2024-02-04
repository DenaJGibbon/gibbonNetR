library(dplyr)
setwd("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")
devtools::document()
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")


# Multi-class models using 'train_CNN_multi' ------------------------------

# Location of spectrogram images for training
input.data.path <-  'data/imagesmalaysiamulti/'

# Location of spectrogram images for testing
test.data.path <- '/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/imagesmalaysiamaliau copy/test/'

# Training data folder short
trainingfolder.short <- 'imagesmalaysiamulti'

# Whether to unfreeze the layers
unfreeze.param <- TRUE # FALSE means the features are frozen; TRUE unfrozen

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Allow early stopping?
early.stop <- 'yes'

gibbonNetR::train_CNN_multi(input.data.path=input.data.path,
                             architecture ='alexnet',
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze = TRUE,
                             epoch.iterations=1,
                             save.model= TRUE,
                             early.stop = "yes",
                             output.base.path = "data/test/",
                             trainingfolder=trainingfolder.short,
                             noise.category = "noise")

gibbonNetR::train_CNN_multi(input.data.path=input.data.path,
                            architecture ='vgg16',
                            learning_rate = 0.001,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            save.model= TRUE,
                            early.stop = "yes",
                            output.base.path = "data/test/",
                            trainingfolder=trainingfolder.short,
                            noise.category = "noise")

gibbonNetR::train_CNN_multi(input.data.path=input.data.path,
                            architecture ='vgg19',
                            learning_rate = 0.001,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            save.model= TRUE,
                            early.stop = "yes",
                            output.base.path = "data/test/",
                            trainingfolder=trainingfolder.short,
                            noise.category = "noise")

gibbonNetR::train_CNN_multi(input.data.path=input.data.path,
                            architecture ='resnet18',
                            learning_rate = 0.001,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            save.model= TRUE,
                            early.stop = "yes",
                            output.base.path = "data/test/",
                            trainingfolder=trainingfolder.short,
                            noise.category = "noise")

gibbonNetR::train_CNN_multi(input.data.path=input.data.path,
                            architecture ='resnet50',
                            learning_rate = 0.001,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            save.model= TRUE,
                            early.stop = "yes",
                            output.base.path = "data/test/",
                            trainingfolder=trainingfolder.short,
                            noise.category = "noise")

gibbonNetR::train_CNN_multi(input.data.path=input.data.path,
                            architecture ='resnet152',
                            learning_rate = 0.001,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            save.model= TRUE,
                            early.stop = "yes",
                            output.base.path = "data/test/",
                            trainingfolder=trainingfolder.short,
                            noise.category = "noise")


performancetables.dir.multi <- '/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/test/_imagesmalaysiamulti_multi_unfrozen_TRUE_/performance_tables_multi'

PerformanceOutputMulti <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir.multi,
                                                      class='duet',
                                                      model.type = "multi")

PerformanceOutputMulti$f1_plot
PerformanceOutputMulti$pr_plot
PerformanceOutputMulti$FPRTPR_plot
PerformanceOutputMulti$best_f1$F1
as.data.frame(PerformanceOutputMulti$best_auc)

# Binary Models using 'train_CNN_binary' ----------------------------------

# Location of spectrogram images for training
input.data.path <-  'data/imagesmalaysia/'

# Location of spectrogram images for testing
test.data.path <- 'data/imagesmalaysiamaliau/test/'

# Training data folder short
trainingfolder.short <- 'imagesmalaysia'

# Whether to unfreeze the layers
unfreeze.param <- TRUE # FALSE means the features are frozen; TRUE unfrozen

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Allow early stopping?
early.stop <- 'yes'

gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                          architecture ='alexnet',
                          learning_rate = 0.001,
                          save.model= TRUE,
                          test.data=test.data.path,
                          unfreeze = TRUE,
                          epoch.iterations=epoch.iterations,
                          early.stop = "yes",
                          output.base.path = "data/test/",
                          trainingfolder=trainingfolder.short,
                          positive.class="Gibbons",
                          negative.class="Noise")

gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='vgg16',
                             learning_rate = 0.001,
                             save.model= TRUE,
                             test.data=test.data.path,
                             unfreeze = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "data/test/",
                             trainingfolder=trainingfolder.short,
                             positive.class="Gibbons",
                             negative.class="Noise")

gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='vgg19',
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             save.model= TRUE,
                             unfreeze = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "data/test/",
                             trainingfolder=trainingfolder.short,
                             positive.class="Gibbons",
                             negative.class="Noise")

gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='resnet18',
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             save.model= TRUE,
                             unfreeze = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "data/test/",
                             trainingfolder=trainingfolder.short,
                             positive.class="Gibbons",
                             negative.class="Noise")

gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='resnet50',
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             save.model= TRUE,
                             unfreeze = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "data/test/",
                             trainingfolder=trainingfolder.short,
                             positive.class="Gibbons",
                             negative.class="Noise")

gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='resnet152',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "data/test/",
                             trainingfolder=trainingfolder.short,
                             positive.class="Gibbons",
                             negative.class="Noise")


performancetables.dir <- '/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/test/_imagesmalaysia_binary_unfrozen_TRUE_/performance_tables/'
PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                      class='Gibbons',
                                                      model.type = "binary")

PerformanceOutput$f1_plot
PerformanceOutput$pr_plot
PerformanceOutput$FPRTPR_plot
PerformanceOutput$best_f1$F1
as.data.frame(PerformanceOutput$best_auc)



# Extract embeddings ------------------------------------------------------

ModelPath <- "/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/test/_imagesmalaysia_binary_unfrozen_TRUE_/_imagesmalaysia_5_alexnet_model.pt"
result <- extract_embeddings(test_input="/Users/denaclink/Desktop/JahooArray/AlexNet27_09_22/Detections/",
                                      model_path=ModelPath,
                                     target_class = "FemaleDetectionsTP")

result$EmbeddingsCombined
result$NMI
result$ConfusionMatrix

# Process spectrogram images for testing data -------------------------------------------------------------------------
# The splits are set to ensure all data (100%) goes into the relevant folder.
gibbonNetR::spectrogram_images(
  trainingBasePath = '/Volumes/DJC Files/Clink et al Zenodo Data/TrainingFilesValidated/', #'/Volumes/DJC Files/Danum Deep Learning/TestClips', #
  outputBasePath   = 'data/imagesmalaysia/',
  splits           = c(1, 0, 0)  # 0% training, 0% validation, 100% testing
)

# The splits are set to ensure all data (100%) goes into the relevant folder.
gibbonNetR::spectrogram_images(
  trainingBasePath = '/Volumes/DJC Files/Clink et al Zenodo Data/ValidationClipsDanum/', #'/Volumes/DJC Files/Danum Deep Learning/TestClips', #
  outputBasePath   = 'data/imagesmalaysia/',
  splits           = c(0, 1, 0)  # 0% training, 0% validation, 100% testing
)


# The splits are set to ensure all data (100%) goes into the relevant folder.
gibbonNetR::spectrogram_images(
  trainingBasePath = '/Volumes/DJC Files/Clink et al Zenodo Data/TestClipsDanum/', #'/Volumes/DJC Files/Danum Deep Learning/TestClips', #
  outputBasePath   = 'data/imagesmalaysia/',
  splits           = c(0, 0, 1)  # 0% training, 0% validation, 100% testing
)



# Ensure no data leakage between train, valid, and test sets --------------
# Load necessary libraries
library(stringr)

# Function to extract the relevant identifier from the filename
extract_file_identifier <- function(filename) {
  components <- str_split_fixed(filename, "_", n = 5)
  identifier <- paste(components[,2], components[,3], components[,4], sep = "_")
  return(identifier)
}

# Retrieve lists of files from the respective folders
trainingDir <- 'data/imagesmalaysia/train'
validationDir <- 'data/imagesmalaysia/valid'
testDir <- 'data/imagesmalaysia/test'

trainFiles <- list.files(trainingDir, pattern = "\\.jpg$", full.names = FALSE, recursive = T)
validationFiles <- list.files(validationDir, pattern = "\\.jpg$", full.names = FALSE, recursive = T)
testFiles <- list.files(testDir, pattern = "\\.jpg$", full.names = FALSE, recursive = T)

# Extract identifiers for each file in the respective datasets
trainIds <- sapply(trainFiles, extract_file_identifier)
validationIds <- sapply(validationFiles, extract_file_identifier)
testIds <- sapply(testFiles, extract_file_identifier)

# Check for data leakage
trainValidationOverlap <- trainIds[which(trainIds %in% validationIds)]
trainTestOverlap <- trainIds[which(trainIds %in% testIds)]
validationTestOverlap <- testIds[which(testIds %in% validationIds)]

# Report findings
if (length(trainValidationOverlap) == 0 & length(trainTestOverlap) == 0 & length(validationTestOverlap) == 0) {
  cat("No data leakage detected among the datasets.\n")
} else {
  cat("Data leakage detected!\n")
  if (length(trainValidationOverlap) > 0) {
    cat("Overlap between training and validation datasets:\n", trainValidationOverlap, "\n")
  }

  if (length(trainTestOverlap) > 0) {
    cat("Overlap between training and test datasets:\n", trainTestOverlap, "\n")
  }

  if (length(validationTestOverlap) > 0) {
    cat("Overlap between validation and test datasets:\n", validationTestOverlap, "\n")
  }
}


# Visualize spectrogram images for training data:-------------------------------------------------------------------------
# Load dplyr package for data manipulation functions
library(dplyr)

# Location of spectrogram images for training
input.data.path <-  'data/imagesmalaysia/'

# Create a dataset of images:
# - The images are sourced from the 'train' subdirectory within the specified path `input.data.path`.
# - The images undergo several transformations:
#     1. They are converted to tensors.
#     2. They are resized to 224x224 pixels.
#     3. Their pixel values are normalized.
train_ds <- image_folder_dataset(
  file.path(input.data.path,'train' ),   # Path to the image directory
  transform = . %>%
    torchvision::transform_to_tensor() %>%
    torchvision::transform_resize(size = c(224, 224)) %>%
    torchvision::transform_normalize(
      mean = c(0.485, 0.456, 0.406),      # Mean for normalization
      std = c(0.229, 0.224, 0.225)        # Standard deviation for normalization
    ),
  target_transform = function(x) as.double(x) - 1  # Transformation for target/labels
)

# Create a dataloader from the dataset:
# - This helps in efficiently loading and batching the data.
# - The batch size is set to 24, with shuffling enabled and the last incomplete batch is dropped.
train_dl <- dataloader(train_ds, batch_size = 24, shuffle = TRUE, drop_last = TRUE)

# Extract the next batch from the dataloader
batch <- train_dl$.iter()$.next()

# Extract the labels for the batch and determine class names
classes <- batch[[2]]
class_names <- ifelse(batch$y, 'Noise','Gibbons')

# Convert the batch tensor of images to an array and process them:
# - The image tensor is permuted to change the dimension order.
# - The pixel values of the images are denormalized.
images <- as_array(batch[[1]]) %>% aperm(perm = c(1, 3, 4, 2))
mean <- c(0.485, 0.456, 0.406)
std <- c(0.229, 0.224, 0.225)
images <- std * images + mean
images <- images * 255
# Clip the pixel values to lie within [0, 255]
images[images > 255] <- 255
images[images < 0] <- 0

# Set the plotting parameters for a 4x6 grid
par(mfcol = c(4,6), mar = rep(1, 4))

# Visualize the images:
# - Use `purrr` functions to handle arrays.
# - Set the name of each image based on its class.
# - Convert each image to a raster format for plotting.
# - Finally, iterate over each image, plotting it and setting its title.
images %>%
  purrr::array_tree(1) %>%
  purrr::set_names(class_names) %>%
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk(~{plot(.x); title(.y)})

# Ensure no data leakage between train, valid, and test sets --------------
# Load necessary libraries
library(stringr)

# Function to extract the relevant identifier from the filename
extract_file_identifier <- function(filename) {
  shortname <- basename(filename)
  identifier <- shortname #paste(components[,1],components[,2], components[,3], components[,4], sep = "_")
  return(identifier)
}

# Retrieve lists of files from the respective folders
trainingDir <- 'data/imagesmalaysiamulti/train'
validationDir <- 'data/imagesmalaysiamulti/valid'
testDir <- 'data/imagesmalaysiamulti/test'

trainFiles <- list.files(trainingDir, pattern = "\\.jpg$", full.names = FALSE, recursive = T)
validationFiles <- list.files(validationDir, pattern = "\\.jpg$", full.names = FALSE, recursive = T)
testFiles <- list.files(testDir, pattern = "\\.jpg$", full.names = FALSE, recursive = T)

# Extract identifiers for each file in the respective datasets
trainIds <- sapply(trainFiles, extract_file_identifier)
validationIds <- sapply(validationFiles, extract_file_identifier)
testIds <- sapply(testFiles, extract_file_identifier)

# Check for data leakage
trainValidationOverlap <- trainIds[which(trainIds %in% validationIds)]
trainTestOverlap <- trainIds[which(trainIds %in% testIds)]
validationTestOverlap <- testIds[which(testIds %in% validationIds)]

# Report findings
if (length(trainValidationOverlap) == 0 & length(trainTestOverlap) == 0 & length(validationTestOverlap) == 0) {
  cat("No data leakage detected among the datasets.\n")
} else {
  cat("Data leakage detected!\n")
  if (length(trainValidationOverlap) > 0) {
    cat("Overlap between training and validation datasets:\n", trainValidationOverlap, "\n")
  }

  if (length(trainTestOverlap) > 0) {
    cat("Overlap between training and test datasets:\n", trainTestOverlap, "\n")
  }

  if (length(validationTestOverlap) > 0) {
    cat("Overlap between validation and test datasets:\n", validationTestOverlap, "\n")
  }
}


# Run trained model over sound directory -------------------------------------------------------------------------
# Predict the test files
ModelPath <- "/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/test/_imagesmalaysiamulti_multi_unfrozen_TRUE_/_imagesmalaysiamulti_1_vgg19_model.pt"
#TopModel <- luz_load(ModelPath)

devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

# Example
deploy_CNN_multi(
  clip_duration = 12,
  architecture='vgg19',
  output_folder = '/Volumes/Clink Data Backup/DanumLocArray/gibbonNetRMulti/',
  output_folder_selections = '/Volumes/Clink Data Backup/DanumLocArray/gibbonNetRMulti/',
  output_folder_wav = '/Volumes/Clink Data Backup/DanumLocArray/gibbonNetRMulti/',
  detect_pattern= c('_070','_080'),
  top_model_path = ModelPath,
  path_to_files = "/Volumes/Clink Data Backup/DanumLocArray/LocalArray/",
  class_names = c('duet','hornbill.helmeted','hornbill.rhino','long.argus','noise'),
  noise_category = 'noise',
  single_class = FALSE,
  save_wav = FALSE,
  threshold = .75
)


