---
title: 'gibbonNetR: an R Package for the Use of Convolutional Neural Networks and Transfer Learning on Passive Acoustic Monitoring (PAM) Data'
tags:
  - deep learning
  - passive acoustic monitoring
  - gibbon
  - automated detection
authors:
  - name: Dena Jane Clink
    affiliation: 1 
  - name: Abdul Hamid Ahmad
    affiliation: 2
affiliations:
 - name: K. Lisa Yang Center for Conservation Bioacoustics,Cornell Lab of Ornithology, Cornell University, Ithaca, New York, United States
   index: 1
 - name: Institute for Tropical Biology and Conservation,Universiti Malaysia Sabah (UMS), Kota Kinabalu, Sabah, Malaysia
   index: 2

date: 2024
bibliography: references.bib

---

# Summary

Automated detection of acoustic signals is crucial for effective monitoring of vocal animals and their habitats across large spatial and temporal scales. Recent advances in deep learning have made high performing automated detection approaches more accessible two more practitioners. However, there are few deep learning approaches that can be implemented natively in R. The 'torch for R' ecosystem has made the use of transfer learning with convolutional neural networks accessible for R users. Here we provide an R package and workflow to use transfer learning for the automated detection of acoustics signals from passive acoustic monitoring (PAM) data collected in Sabah, Malaysia. The package provides functions to create spectogram images from PAM data, compare the performance of different pre-trained CNN architectures, and deploy trained models over directories of sound files. The R programming language remains one of the most commonly used languages among ecologists, and we hope that this package makes deep learning approaches more accessible to this audience.  

# Statement of need

## *Passive acoustic monitoring*

We are in a biodiversity crisis, and there is a great need for the ability to rapidly assess biodiversity in order to understand and mitigate anthropogenic impacts. One approach that can be especially effective for monitoring of vocal yet cryptic animals is the use of passive acoustic monitoring [@gibb2018], a technique that relies autonomous acoustic recording units. PAM allows researchers to monitor vocal animals and their habitats, at temporal and spatial scales that are impossible to achieve using only human observers. Interest in use of PAM in terrestrial environments has increased substantially in recent years [@sugai2019], due to reduced price of the recording units and improved battery life and data storage capabilities. However, the use of PAM often leads to the collection of terabytes of data that is time- and cost-prohibitive to analyze manually.

## *Automated detection*

Some commonly used non-deep learning approaches for the automated detection of acoustic signals in terrestrial PAM data include binary point matching [@katz2016], spectrogram cross-correlation [@balantic2020], or the use of a band- limited energy detector and subsequent classifier, such as support vector machine [@clink2023; @kalan2015]. Recent advances in deep learning have revolutionized image and speech recognition [@lecun2015 ], with important cross-over for the analysis of PAM data. Traditional approaches to machine learning relied heavily on feature engineering, as early machine learning algorithms required a reduced set of representative features, such as features estimated from the spectrogram. Deep learning does not require feature engineering [@stevens2020] . Convolutional neural networks (CNNs) --- one of the most effective deep learning algorithms---are useful for processing data that have a 'grid-like topology', such as image data that can be considered a 2-dimensional grid of pixels [@goodfellow2016]. The 'convolutional' layer learns the feature representations of the inputs; these convolutional layers consist of a set of filters which are basically two-dimensional matrices of numbers and the primary parameter is the number of filters [@gu2018]. Therefore, with CNN's there is no feature engineering required. However, if training data are scarce, overfitting may occur as representations of images tend to be large with many variables [@lecun1995].

# *Transfer learning?*
Transfer learning is an approach wherein the architecture of a pretrained CNN (which is generally trained on a very large dataset) is applied to a new classification problem. For example, CNNs trained on the ImageNet dataset of \> 1 million images [@deng2009]such as ResNet have been applied to automated detection/classification of primate and bird species from PAM data [@dufourq2022; @ruan2022]. At the most basic level, transfer learning in computer vision applications retains the feature extraction or embedding layers, and modifies the last few classification layers to be trained for a new classification task [@dufourq2022].

## *'torch for R' ecosystem*

'Keras' [@chollet2015], 'PyTorch' [@paszke2019] and 'Tensorflow' [@martínabadi2015] are some of the more popular neural network libraries; these libraries were all initially developed for the Python programming language. Until recently, deep learning implementations in R relied on the 'reticulate' package which served as an interface to Python [@ushey2022]. However, the recent release of the 'torch for R' ecosystem provides a framework based on 'PyTorch' that runs natively in R and has no dependency on Python [@falbel2023]. Running natively in R means more straightforward installation, and higher accessibility for users of the R programming environment. @keydana2023 provides tutorials for transfer learning in the 'torch for R' ecosystem, and the functions in 'gibbonNetR' rely heavily on these tutorials.

# Overview

This package provides functions to create spectrogram images, use transfer learning from six pretrained CNN architectures (AlexNet [@krizhevsky2017] , VGG16, VGG19 [@simonyan2014], ResNet18, ResNet50, and ResNet152 [@he2016]), evaluate model performance, deploy the highest performing model over a directory of sound files, and extract embeddings from trained models to visualize acoustic data. We provide an example dataset that consists of labelled vocalizations of the loud calls of four vertebrates from Danum Valley Conservation Area, Sabah, Malaysia.

# Usage

```{r,echo=FALSE, message=FALSE, warning=FALSE}
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

```

## First we create spectrogram images

```{r, echo=FALSE, out.width="75%",fig.cap='Spectrograms of training clips for CNNs',fig.align='center',fig.pos = "H"}
knitr::include_graphics("README_files/spectro.png")
```

## Then we train the model

# Train the models

## Training the models using gibbonNetR and evaluating on a test set

```{r,eval = FALSE}
# Location of spectrogram images for training
input.data.path <-  'data/examples/'

# Location of spectrogram images for testing
test.data.path <- 'data/examples/test/'

# User specified training data label for metadata
trainingfolder.short <- 'danummulticlassexample'

# We can specify the number of epochs to train here
epoch.iterations <- c(20)

# Function to train a multi-class CNN
gibbonNetR::train_CNN_multi(input.data.path=input.data.path,
                            architecture ='resnet50',
                            learning_rate = 0.001,
                            class_weights = c(0.3, 0.3, 0.2, 0.2, 0),
                            test.data=test.data.path,
                            unfreeze.param = TRUE,
                            epoch.iterations=epoch.iterations,
                            save.model= TRUE,
                            early.stop = "yes",
                            output.base.path = "model_output/",
                            trainingfolder=trainingfolder.short,
                            noise.category = "noise")

```

# Evaluating model performance

## Specify for the 'female.gibbon' class

```{r, eval = FALSE, warning=FALSE, results='hide'}
# Evaluate model performance
performancetables.dir <- "/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/model_output/_danummulticlassexample_multi_unfrozen_TRUE_/performance_tables_multi"

PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                      class='female.gibbon',
                                                      model.type = "multi",Thresh.val=0)

```

## Examine the results

```{r, eval = FALSE, warning=FALSE}
PerformanceOutput$f1_plot
PerformanceOutput$best_f1$F1
```

## Specify for the 'hornbill.helmeted' class

```{r, eval = FALSE, warning=FALSE, results='hide'}

# Evaluate model performance
performancetables.dir <- "/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/model_output/_danummulticlassexample_multi_unfrozen_TRUE_/performance_tables_multi"

PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                      class='hornbill.helmeted',
                                                      model.type = "multi",Thresh.val=0)

```

## Examine the results

```{r,eval = FALSE, warning=FALSE}
PerformanceOutput$f1_plot
PerformanceOutput$best_f1$F1
```

# Use the pre-trained model to extract embeddings and use unsupervised clustering to identify signals

## Extract embeddings

```{r, eval = FALSE, warning=FALSE, results='hide'}

ModelPath <- "/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/model_output/_danummulticlassexample_multi_unfrozen_TRUE_/_danummulticlassexample_20_resnet50_model.pt"
result <- extract_embeddings(test_input="/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/examples/test/",
                                      model_path=ModelPath,
                                     target_class = "female.gibbon")
```

## We can plot the unsupervised clustering results

```{r, eval = FALSE}
result$EmbeddingsCombined
```

### We can output the NMI results, and the confusion matrix results when we use 'hdbscan' to match the target class to the cluster with the largest number of observations

```{r,eval = FALSE}
result$NMI
result$ConfusionMatrix
```
# References 




