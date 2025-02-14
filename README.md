gibbonNetR R Package
================
Dena J. Clink and Abdul Hamid Ahmad
2025-02-14

# Overview

This README provides code for training, testing, and deploying,
different convolutional neural network model architectures for automated
detection and classification of acoustic data.

Users can train both binary and multi-class classification models on
spectrogram images, and evaluate their performance on test datasets. The
package includes tools for performance evaluation, allowing easy
identification of the best-performing model. Once the best-performing
model is identified, it can be deployed for large-scale inference on
multiple sound files. In addition to classification, trained CNNs can be
used as feature extractors. Combined with unsupervised clustering, this
enables visualization of differences in acoustic signals.

# Usage

A detailed usage guide can be found at:
<https://denajgibbon.github.io/gibbonNetR/>

# Prerequisites

This package assumes a basic understanding of machine learning, deep
learning, and convolutional neural networks (CNNs). Users should be
familiar with training and evaluating models, as well as handling
spectrogram image data. For those new to these concepts, we recommend
reviewing foundational machine learning and deep learning resources
before using this package. a good starting point would be:

Stowell, Dan. “Computational bioacoustics with deep learning: a review
and roadmap.” PeerJ 10 (2022): e13152.
<https://peerj.com/articles/13152/>

Some practical information on improving model performance can be found
here:
<https://kahst.github.io/BirdNET-Analyzer/best-practices/training.html#>

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

# Set file path to spectrogram images
filepath <- system.file("extdata", "multiclass/", package = "gibbonNetR")

# Train simple CNN model
train_CNN_multi(
  input.data.path = filepath,
  test.data = paste(filepath, "/test/", sep = ""),
  architecture = "alexnet", # Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'
  unfreeze.param = TRUE,
  batch_size = 6,
  class_weights = rep((1 / 5), 5),
  learning_rate = 0.001,
  epoch.iterations = 1, # Or any other list of integer epochs
  early.stop = "yes",
  save.model = FALSE,
  output.base.path = file.path(tempdir(), "/MultiDir/", sep = ""),
  trainingfolder = "test_multi",
  noise.category = "noise"
)
```
