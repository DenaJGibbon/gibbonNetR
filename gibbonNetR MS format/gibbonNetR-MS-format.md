---
title: "gibbonNetR: an R Package for the Use of Convolutional Neural Networks and Transfer Learning on Acoustic Data"
preprint: false
author: 
  - name: Dena Jane Clink
    affiliation: 1
    corresponding: true
    email: dena.clink@cornell.edu
  - name: Abdul Hamid Ahmad
    affiliation: 2
affiliation:
  - code: 1
    address: K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology, Cornell University, Ithaca, New York, United States
  - code: 2
    address: Institute for Tropical Biology and Conservation, Universiti Malaysia Sabah (UMS), Kota Kinabalu, Sabah, Malaysia
abstract: >
  Automated detection of acoustic signals is crucial for effective monitoring of vocal animals and their habitats across large spatial and temporal scales. Recent advances in deep learning have made high performing automated detection approaches more accessible two more practitioners. However, there are few deep learning approaches that can be implemented natively in R. The 'torch for R' ecosystem has made the use of transfer learning with convolutional neural networks possible for R users. Here we provide an R package and workflow to use transfer learning for the automated detection of acoustics signals from passive acoustic monitoring (PAM) data collected in Sabah, Malaysia. The package provides functions to create spectogram images from PAM data, compare the performance of different pre-trained CNN architectures, and deploy trained models over directories of sound files. 
bibliography: references.bib
output:
  rticles::joss_article: default
  bookdown::pdf_book:
    base_format: rticles::joss_article # for using bookdown features like \@ref()
---

# Introduction

## *Passive acoustic monitoring*

We are in a biodiversity crisis, and there is a great need for the ability to rapidly assess biodiversity in order to understand and mitigate anthropogenic impacts. One approach that can be especially effective for monitoring of vocal yet cryptic animals is the use of passive acoustic monitoring [@gibb2018], a technique that relies autonomous acoustic recording units. PAM allows researchers to monitor vocal animals and their habitats, at temporal and spatial scales that are impossible to achieve using only human observers. Interest in use of PAM in terrestrial environments has increased substantially in recent years [@sugai2019], due to reduced price of the recording units and improved battery life and data storage capabilities. However, the use of PAM often leads to the collection of terabytes of data that is time- and cost-prohibitive to analyze manually.

## *Automated detection*

Some commonly used non-deep learning approaches for the automated detection of acoustic signals in terrestrial PAM data include binary point matching [@katz2016], spectrogram cross-correlation [@balantic2020], or the use of a band- limited energy detector and subsequent classifier, such as support vector machine [@clink2023; @kalan2015]. Recent advances in deep learning have revolutionized image and speech recognition [@lecun2015 ], with important cross-over for the analysis of PAM data.

## *'torch for R ecosystem'*

The two most popular open-source programming languages are R and Python [@scavetta2021 ]. Python has surpassed R in terms of overall popularity, but R remains an important language for the life sciences [@lawlor2022]. 'Keras' [@chollet2015], 'PyTorch' [@paszke2019] and 'Tensorflow' [@martínabadi2015] are some of the more popular neural network libraries; these libraries were all initially developed for the Python programming language. Until recently, deep learning implementations in R relied on the 'reticulate' package which served as an interface to Python [@ushey2022]. However, the recent release of the 'torch for R' ecosystem provides a framework based on 'PyTorch' that runs natively in R and has no dependency on Python [ @falbel2023]. Running natively in R means more straightforward installation, and higher accessibility for users of the R programming environment.

# Materials and methods

\begin{figure}[ht]

{\centering \includegraphics[width=0.5\linewidth]{../README_files/spectro} 

}

\caption{Spectrograms of training clips for CNNs}\label{fig:unnamed-chunk-1}
\end{figure}

# Results {.unnumbered}

\begin{figure}[ht]

\includegraphics{gibbonNetR-MS-format_files/figure-latex/unnamed-chunk-2-1} \hfill{}

\caption{Evaluating performance of pretrained CNNs}\label{fig:unnamed-chunk-2}
\end{figure}

# Discussion {.unnumbered}

# Acknowledgments {.unnumbered}

So long and thanks for all the fish.

# References
