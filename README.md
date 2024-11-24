gibbonNetR: R Package for the Use of CNNs and Transfer Learning on
Acoustic Data
================
Dena J. Clink and Abdul Hamid Ahmad
2024-11-24

# Overview

This README provides code for training and testing the performance of
different convolutional neural network model architectures on
spectrogram images.

# Usage

A detailed usage guide can be found at:
denajgibbon.github.io/gibbonNetR/

# Installation

You can install the `gibbonNetR` package from its repository using
`devtools`:

``` r
# If you don't have devtools installed
install.packages("devtools")

# Install gibbonNetR
devtools::install_github("https://github.com/DenaJGibbon/gibbonNetR")

# The first time you use the package 'torch' will need to install additional packages. You can start the process using the following:
library(torch)
```

# Quickstart guide

``` r
  library(gibbonNetR)

  # Set file path to spectorgram images  
  filepath <- system.file("extdata", "multiclass/", package = "gibbonNetR")

  # Train simple CNN model
  train_CNN_multi(
    input.data.path = filepath,
    test.data = paste(filepath,'/test/',sep=''),
    architecture = "alexnet",  # Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'
    unfreeze.param = TRUE,
    batch_size = 6,
    class_weights = rep( (1/5), 5),
    learning_rate = 0.001,
    epoch.iterations = 1,  # Or any other list of integer epochs
    early.stop = "yes",
    save.model= FALSE,
    output.base.path = paste(tempdir(),'/MultiDir/',sep=''),
    trainingfolder = "test_multi",
    noise.category = 'noise'
  )
```
