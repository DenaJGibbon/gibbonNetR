library(dplyr)
setwd("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")
devtools::document()
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")



# Process spectrogram images for testing data:-------------------------------------------------------------------------
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


# Get setup for binary training --------------------------------------------------
setwd("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")
devtools::load_all()

# Location of spectrogram images for training
input.data.path <-  'data/imagesmalaysia/'

# Location of spectrogram images for testing
test.data.path <- 'data/imagesmalaysia/'

# Training data folder short
trainingfolder.short <- 'imagesmalaysia'

# Whether to unfreeze the layers
unfreeze.param <- FALSE # FALSE means the features are frozen; TRUE unfrozen

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5)

# Location to save the out
output.data.path <-paste('data/','output','unfrozen',unfreeze.param,trainingfolder.short,'/', sep='_')

# Create if doesn't exist
dir.create(output.data.path)

# Allow early stopping?
early.stop <- 'yes' # NOTE: Must comment out if don't want early stopping

gibbonNetR::train_alexNet(input.data.path=input.data.path,
                          test.data=test.data.path,
                          unfreeze = TRUE,
                          epoch.iterations=epoch.iterations,
                          early.stop = "yes",
                          output.base.path = "data/",
                          trainingfolder=trainingfolder.short,
                          positive.class="Gibbons",
                          negative.class="Noise")


gibbonNetR::train_VGG16(input.data.path=input.data.path,
                          test.data=test.data.path,
                          unfreeze = TRUE,
                          epoch.iterations=epoch.iterations,
                          early.stop = "yes",
                          output.base.path = "data/",
                          trainingfolder=trainingfolder.short,
                          positive.class="Gibbons",
                          negative.class="Noise")

gibbonNetR::train_VGG19(input.data.path=input.data.path,
                        test.data=test.data.path,
                        unfreeze = TRUE,
                        epoch.iterations=epoch.iterations,
                        early.stop = "yes",
                        output.base.path = "data/",
                        trainingfolder=trainingfolder.short,
                        positive.class="Gibbons",
                        negative.class="Noise")

gibbonNetR::train_ResNet18(input.data.path=input.data.path,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            early.stop = "yes",
                            output.base.path = "data/",
                            trainingfolder=trainingfolder.short,
                            positive.class="Gibbons",
                            negative.class="Noise")

gibbonNetR::train_ResNet50(input.data.path=input.data.path,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            early.stop = "yes",
                            output.base.path = "data/",
                            trainingfolder=trainingfolder.short,
                            positive.class="Gibbons",
                            negative.class="Noise")

gibbonNetR::train_ResNet152(input.data.path=input.data.path,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            early.stop = "yes",
                            output.base.path = "data/",
                            trainingfolder=trainingfolder.short,
                            positive.class="Gibbons",
                            negative.class="Noise")

performancetables.dir <- '/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/_output_unfrozen_TRUE_imagesmalaysia_/performance_tables/'
PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir)

PerformanceOutput$f1_plot
PerformanceOutput$pr_plot
PerformanceOutput$FPRTPR_plot
PerformanceOutput$best_f1$F1


# Visualize spectrograms for testing data ---------------------------------
# The splits are set to ensure all data (100%) goes into the relevant folder.
gibbonNetR::spectrogram_images(
  trainingBasePath = '/Volumes/DJC Files/Clink et al Zenodo Data/TestClipsMaliau/', #
  outputBasePath   = 'data/imagesmalaysiamaliau/',
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
testDir <- 'data/imagesmalaysiamaliau/test'

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

# Location of spectrogram images
input.data.path <-  'data/imagesmalaysiamaliau/'

# Create a dataset of images:
# - The images are sourced from the 'train' subdirectory within the specified path `input.data.path`.
# - The images undergo several transformations:
#     1. They are converted to tensors.
#     2. They are resized to 224x224 pixels.
#     3. Their pixel values are normalized.
train_ds <- image_folder_dataset(
  file.path(input.data.path,'test' ),   # Path to the image directory
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



trained_models_dir <- '/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/_output_unfrozen_TRUE_imagesmalaysia_'

#image_data_dir <- '/Volumes/DJC 1TB/VocalIndividualityClips/RandomSelectionImages/'
image_data_dir <- 'data/imagesmalaysiamaliau/test/'

evaluate_trainedmodel_performance(trained_models_dir=trained_models_dir,
                                  image_data_dir=image_data_dir)


PerformanceOutPutTrained <- gibbonNetR::get_best_performance(performancetables.dir='data/performance_tables_trained/')

PerformanceOutPutTrained$f1_plot
PerformanceOutPutTrained$best_f1$F1
PerformanceOutPutTrained$pr_plot
# PerformanceOutPutTrained$FPRTPR_plot
# PerformanceOutPutTrained$best_auc$AUC



# Multi-class -------------------------------------------------------------

MultiClipPath <- '/Volumes/DJC Files/Danum Deep Learning/MultiSpeciesAnalysis/TrainingClipsMulti'

gibbonNetR::spectrogram_images(
  trainingBasePath = MultiClipPath,
  outputBasePath   = 'data/imagesmalaysiamulti/',
  splits           = c(.6, .2, .2)  # 0% training, 0% validation, 100% testing
)



# Ensure no data leakage between train, valid, and test sets --------------
# Load necessary libraries
library(stringr)

# Function to extract the relevant identifier from the filename
extract_file_identifier <- function(filename) {
  shortname <- basename(filename)
  components <- str_split_fixed(shortname, "_", n = 5)
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


# Get setup for training --------------------------------------------------
setwd("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")
devtools::load_all()

# Location of spectrogram images for training
input.data.path <-  'data/imagesmalaysiamulti/'

# Location of spectrogram images for testing
test.data.path <- 'data/imagesmalaysiamulti/'

# Training data folder short
trainingfolder.short <- 'imagesmalaysiamulti'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Location to save the out
output.data.path <-paste('data/multi/','output','unfrozen',TRUE,trainingfolder.short,'/', sep='_')

# Create if doesn't exist
dir.create(output.data.path)

# Allow early stopping?
early.stop <- 'yes' # NOTE: Must comment out if don't want early stopping

  train_alexNet_multiClass(
    input.data.path = input.data.path,
    test.data = test.data.path,
    unfreeze = TRUE,
    epoch.iterations = epoch.iterations,
    early.stop = early.stop,
    output.base.path = output.data.path,
    trainingfolder = trainingfolder.short
  )

  performancetables.dir <-"/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/multi/_output_unfrozen_TRUE_imagesmalaysiamulti_/performance_tables/"
  PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                        class='hornbill.helmeted')

  PerformanceOutput$f1_plot

  train_VGG16_multiClass(
    input.data.path = input.data.path,
    test.data = test.data.path,
    unfreeze = TRUE,
    epoch.iterations = epoch.iterations,
    early.stop = early.stop,
    output.base.path = output.data.path,
    trainingfolder = trainingfolder.short
  )

  performancetables.dir <-"/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/multi/_output_unfrozen_TRUE_imagesmalaysiamulti_/performance_tables/"
  PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                        class='duet')

  PerformanceOutput$f1_plot

  train_VGG19_multiClass(
    input.data.path = input.data.path,
    test.data = test.data.path,
    unfreeze = TRUE,
    epoch.iterations = epoch.iterations,
    early.stop = early.stop,
    output.base.path = output.data.path,
    trainingfolder = trainingfolder.short
  )

  performancetables.dir <-"/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/multi/_output_unfrozen_TRUE_imagesmalaysiamulti_/performance_tables/"
  PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                        class='hornbill.helmeted')

  PerformanceOutput$f1_plot

  train_ResNet18_multiClass(
    input.data.path = input.data.path,
    test.data = test.data.path,
    unfreeze = TRUE,
    epoch.iterations = epoch.iterations,
    early.stop = early.stop,
    output.base.path = output.data.path,
    trainingfolder = trainingfolder.short
  )

  performancetables.dir <-"/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/multi/_output_unfrozen_TRUE_imagesmalaysiamulti_/performance_tables/"
  PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                        class='hornbill.helmeted')

  PerformanceOutput$f1_plot

  train_ResNet50_multiClass(
    input.data.path = input.data.path,
    test.data = test.data.path,
    unfreeze = TRUE,
    epoch.iterations = epoch.iterations,
    early.stop = early.stop,
    output.base.path = output.data.path,
    trainingfolder = trainingfolder.short
  )

  performancetables.dir <-"/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/multi/_output_unfrozen_TRUE_imagesmalaysiamulti_/performance_tables/"
  PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                        class='hornbill.helmeted')

  PerformanceOutput$f1_plot

  train_ResNet152_multiClass(
    input.data.path = input.data.path,
    test.data = test.data.path,
    unfreeze = TRUE,
    epoch.iterations = 1,
    early.stop = early.stop,
    output.base.path = output.data.path,
    trainingfolder = trainingfolder.short
  )

  performancetables.dir <-"/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/multi/_output_unfrozen_TRUE_imagesmalaysiamulti_/performance_tables/"
  PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                        class='long.argus')

  PerformanceOutput$f1_plot
  PerformanceOutput$pr_plot
  PerformanceOutput$FPRTPR_plot
  PerformanceOutput$best_f1$F1
  PerformanceOutput$best_auc


# Multi Maliau Test -------------------------------------------------------

  # note had to manually move some incorrectly classified images
  # The splits are set to ensure all data (100%) goes into the relevant folder.
  gibbonNetR::spectrogram_images(
    trainingBasePath = '/Volumes/DJC Files/Danum Deep Learning/MultiSpeciesAnalysis/ValidationClipsMaliau/', #'/Volumes/DJC Files/Danum Deep Learning/TestClips', #
    outputBasePath   = 'data/imagesmalaysiamaliaumulti/',
    splits           = c(0, 0, 1)  # 0% training, 0% validation, 100% testing
  )


  trained_models_dir <- "data/multi/_output_unfrozen_TRUE_imagesmalaysiamulti_/"

  #image_data_dir <- '/Volumes/DJC 1TB/VocalIndividualityClips/RandomSelectionImages/'
  image_data_dir <- 'data/imagesmalaysiamaliaumulti/test/'

  evaluate_trainedmodel_performance_multi(trained_models_dir=trained_models_dir,
                                    image_data_dir=image_data_dir,
                                    output_dir='data/')


  PerformanceOutPutTrained <- gibbonNetR::get_best_performance(performancetables.dir='data/performance_tables_multi_trained/',
                                                               class='hornbill.rhino')

  PerformanceOutPutTrained$f1_plot
  PerformanceOutPutTrained$best_f1$F1
  PerformanceOutPutTrained$pr_plot
