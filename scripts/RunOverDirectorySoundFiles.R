# Prepare data ------------------------------------------------------------
setwd("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")
devtools::document()
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

# NEED TO ADD:  metadata output

# Set the output folder paths
OutputFolder <- '/Volumes/Clink Data Backup/DanumLocArray/HelmetedHornbill/Detections/'
OutputFolderSelections <- '/Volumes/Clink Data Backup/DanumLocArray/HelmetedHornbill/Detections/HornbillSelectionTables/'
OutputFolderWav <- '/Volumes/Clink Data Backup/DanumLocArray/HelmetedHornbill/Detections/HornbillWavs/'

# Import trained model
# Chose because had highest precision while maintaining recall
TopModel <- luz_load("data/multi/_output_unfrozen_TRUE_imagesmalaysiamulti_/_imagesmalaysiamulti_4_modelalexNet.pt")

# path.to.files <- '/Users/denaclink/Library/CloudStorage/Box-Box/Cambodia 2022/Acoustic Gibbon PAM 27_09_22' # Do not run this again because T/F positives split

# path.to.files <- '/Users/denaclink/Library/CloudStorage/Box-Box/Cambodia 2022/Acoustics Gibbon PAM 15_03_22'

path.to.files <- '/Volumes/Clink Data Backup/DanumLocArray/20180222 to 20180404 SW 6 to 10/'
# Already run above this line #

#
# path.to.files <-'/Users/denaclink/Library/CloudStorage/Box-Box/Cambodia 2022/Acoustics Gibbon PAM_04_03_23'
# Doesn't have all 10 #/Users/denaclink/Library/CloudStorage/Box-Box/Cambodia 2022/Acoustics Gibbon PAM_08_07_23'

# Get a list of full file paths of sound files in a directory
SoundFilePathFull <- list.files(path.to.files,
                                recursive = T, pattern = '.wav', full.names = T)

# Get a list of file names of sound files in a directory
SoundFilePathShort <- list.files(path.to.files,
                                 recursive = T, pattern = '.wav')

# Extract only the file names without the extension
SoundFilePathShort <- str_split_fixed(SoundFilePathShort,
                                      pattern = '.wav', n = 2)[,1]

# Count the number of slashes in the first file path
nslash <- str_count(SoundFilePathShort[1], '/') + 1

# Split the file paths based on slashes and keep the last part (file name)
SoundFilePathShort <- str_split_fixed(SoundFilePathShort,
                                      pattern = '/', n = nslash)[, nslash]

# Extract the times from the file names
times <- as.numeric(substr(str_split_fixed(SoundFilePathShort, pattern = '_', n = 3)[, 3], 1, 2))

# Select the indices of times that are between 6 and 10
times.index <- which(times == 6 | times == 7 | times == 8 | times == 9 | times == 10 |
                       times == 11| times == 12 | times == 13 | times == 14 | times == 15 |
                       times == 16 | times == 17 | times == 18)

# Filter the full file paths and file names based on the selected indices
SoundFilePathFull <- SoundFilePathFull[times.index]
SoundFilePathShort <- SoundFilePathShort[times.index]

# Set parameters for processing sound clips
clip.duration <- 12       # Duration of each sound clip
hop.size <- 6             # Hop size for splitting the sound clips
downsample.rate <- 'NA'   # Downsample rate for audio in Hz, otherwise NA
threshold <- 0.85         # Threshold for audio detection
sav.wav <- T              # Save the extracted sound clips as WAV files?
UniqueClassesTraining <- c('duet','hornbill.helmeted','hornbill.rhino','long.argus','noise')
noise.category <- 'noise'

# Create output folders if they don't exist
dir.create(OutputFolder, recursive = TRUE, showWarnings = FALSE)
dir.create(OutputFolderSelections, recursive = TRUE, showWarnings = FALSE)
dir.create(OutputFolderWav, recursive = TRUE, showWarnings = FALSE)

for(x in (1:length(SoundFilePathFull)) ){ tryCatch({
  RavenSelectionTableDFAlexNet <- data.frame()

  start.time.detection <- Sys.time()
  print(paste(x, 'out of', length(SoundFilePathFull)))
  TempWav <- readWave(SoundFilePathFull[x])
  WavDur <- duration(TempWav)

  Seq.start <- list()
  Seq.end <- list()

  i <- 1
  while (i + clip.duration < WavDur) {
    # print(i)
    Seq.start[[i]] = i
    Seq.end[[i]] = i+clip.duration
    i= i+hop.size
  }


  ClipStart <- unlist(Seq.start)
  ClipEnd <- unlist(Seq.end)

  TempClips <- cbind.data.frame(ClipStart,ClipEnd)



  # Subset sound clips for classification -----------------------------------
  print('saving sound clips')
  set.seed(13)
  length <- nrow(TempClips)

  if(length > 100){
    length.files <- seq(1,length,100)
  } else {
    length.files <- c(1,length)
  }

  for(q in 1: (length(length.files)-1) ){
    unlink('/Users/denaclink/Desktop/datacopy/Temp/WavFiles', recursive = TRUE)
    unlink('/Users/denaclink/Desktop/datacopy/Temp/Images/Images', recursive = TRUE)

    dir.create('/Users/denaclink/Desktop/datacopy/Temp/WavFiles')
    dir.create('/Users/denaclink/Desktop/datacopy/Temp/Images/Images')

    RandomSub <-  seq(length.files[q],length.files[q+1],1)

    if(q== (length(length.files)-1) ){
      RandomSub <-  seq(length.files[q],length,1)
    }

    start.time <- TempClips$ClipStart[RandomSub]
    end.time <- TempClips$ClipEnd[RandomSub]

    short.sound.files <- lapply(1:length(start.time),
                                function(i)
                                  extractWave(
                                    TempWav,
                                    from = start.time[i],
                                    to = end.time[i],
                                    xunit = c("time"),
                                    plot = F,
                                    output = "Wave"
                                  ))

    if(downsample.rate != 'NA'){
      print('downsampling')
      short.sound.files <- lapply(1:length(short.sound.files),
                                  function(i)
                                    downsample(
                                      short.sound.files[[i]],
                                      samp.rate=downsample.rate
                                    ))
    }

    for(d in 1:length(short.sound.files)){
      #print(d)
      writeWave(short.sound.files[[d]],paste('/Users/denaclink/Desktop/datacopy/Temp/WavFiles','/',
                                             SoundFilePathShort[x],'_',start.time[d], '.wav', sep=''),
                extensible = F)
    }

    # Save images to a temp folder
    print(paste('Creating images',start.time[1],'start time clips'))

    for(e in 1:length(short.sound.files)){
      jpeg(paste('/Users/denaclink/Desktop/datacopy/Temp/Images/Images','/', SoundFilePathShort[x],'_',start.time[e],'.jpg',sep=''),res = 50)
      short.wav <- short.sound.files[[e]]

      seewave::spectro(short.wav,tlab='',flab='',axisX=F,axisY = F,scale=F,grid=F,flim=c(0.4,3),fastdisp=TRUE,noisereduction=1)

      graphics.off()
    }

    # Predict using AlexNet ----------------------------------------------------
    print('Classifying images using Top Model')

    test.input <- '/Users/denaclink/Desktop/datacopy/Temp/Images/'

    # ResNet
    test_ds <- image_folder_dataset(
      file.path(test.input),
      transform = . %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_resize(size = c(224, 224)) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)))

    # AlexNet and VGG19
    # test_ds <- image_folder_dataset(
    #    file.path(test.input ),
    #    transform = . %>%
    #      torchvision::transform_to_tensor() %>%
    #      torchvision::transform_color_jitter() %>%
    #      transform_resize(256) %>%
    #      transform_center_crop(224) %>%
    #      transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)), target_transform = function(x) as.double(x) - 1 )
    #

    # Predict the test files
    # Variable indicating the number of files

    # Load the test images
    test_dl <- dataloader(test_ds, batch_size =32, shuffle = F)

    # Predict using TopModel
    TopModelPred <- predict(TopModel, test_dl)

    # Return the index of the max values (i.e. which class)
    PredMPS <- torch_argmax(TopModelPred, dim = 2)

    # Save to cpu
    PredMPS <- as_array(torch_tensor(PredMPS, device = 'cpu'))

    # Convert to a factor
    modelResnetPred <- as.factor(PredMPS)

    # Calculate the probability associated with each class
    Probability <- as_array(torch_tensor(nnf_softmax(TopModelPred, dim = 2), device = 'cpu'))

    # Find the index of the maximum value in each row
    max_prob_idx <- apply(Probability, 1, which.max)

    # Map the index to actual probability
    predicted_class_probability <- sapply(1:nrow(Probability), function(i) Probability[i, max_prob_idx[i]])

    # Convert the integer predictions to factor and then to character based on the levels
    modelResnetNames <- factor(modelResnetPred, levels = 1:length(UniqueClassesTraining), labels = UniqueClassesTraining)

    outputTableTopModel <- cbind.data.frame(modelResnetNames, predicted_class_probability)
    colnames(outputTableTopModel) <- c('PredictedClass', 'Probability')

    image.files <- list.files(file.path(test.input),recursive = T,
                              full.names = T)
    nslash <- str_count(image.files,'/')+1
    nslash <- nslash[1]
    image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
    image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]

    print('Saving output')

    Detections <-  which(outputTableTopModel$Probability >= threshold & outputTableTopModel$PredictedClass != noise.category )

    Detections <-  split(Detections, cumsum(c(
      1, diff(Detections)) != 1))

    for(i in 1:length(Detections)){
      TempList <- Detections[[i]]
      if(length(TempList)==1){
        Detections[[i]] <- TempList[1]
      }
      if(length(TempList)==2){
        Detections[[i]] <- TempList[2]
      }
      if(length(TempList)> 2){
        Detections[[i]] <- median(TempList)
      }

    }

    DetectionIndices <- unname(unlist(Detections))

    DetectionClass <-  outputTableTopModel$PredictedClass[DetectionIndices]
    print('Saving output')
    file.copy(image.files[DetectionIndices],
              to= paste(OutputFolder, DetectionClass,'_',
                        image.files.short[DetectionIndices],
                        '_',
                        round(Probability[DetectionIndices],2),
                        '_AlexNet_.jpg', sep=''))

    if(sav.wav ==T){
      wav.file.paths <- list.files('/Users/denaclink/Desktop/datacopy/Temp/WavFiles',full.names = T)
      file.copy(wav.file.paths[DetectionIndices],
                to= paste(OutputFolderWav,  DetectionClass,'_',
                          image.files.short[DetectionIndices],
                          '_',
                          round(Probability[DetectionIndices],2),
                          '_AlexNet_.wav', sep=''))
    }

    Detections <- image.files.short[DetectionIndices]


    if (length(Detections) > 0) {
      Selection <- seq(1, length(Detections))
      View <- rep('Spectrogram 1', length(Detections))
      Channel <- rep(1, length(Detections))
      MinFreq <- rep(100, length(Detections))
      MaxFreq <- rep(2000, length(Detections))
      start.time.new <- as.numeric(str_split_fixed(Detections,pattern = '_',n=4)[,4])
      end.time.new <- start.time.new + clip.duration
      Probability <- round(Probability[DetectionIndices],2)

      RavenSelectionTableDFAlexNetTemp <-
        cbind.data.frame(Selection,
                         View,
                         Channel,
                         MinFreq,
                         MaxFreq,start.time.new,end.time.new,Probability,
                         Detections)

      RavenSelectionTableDFAlexNetTemp <-
        RavenSelectionTableDFAlexNetTemp[, c(
          "Selection",
          "View",
          "Channel",
          "start.time.new",
          "end.time.new",
          "MinFreq",
          "MaxFreq",
          'Probability',"Detections"
        )]

      colnames(RavenSelectionTableDFAlexNetTemp) <-
        c(
          "Selection",
          "View",
          "Channel",
          "Begin Time (s)",
          "End Time (s)",
          "Low Freq (Hz)",
          "High Freq (Hz)",
          'Probability',
          "Detections"
        )

      RavenSelectionTableDFAlexNet <- rbind.data.frame(RavenSelectionTableDFAlexNet,
                                                       RavenSelectionTableDFAlexNetTemp)

      if(nrow(RavenSelectionTableDFAlexNet) > 0){
        csv.file.name <-
          paste(OutputFolderSelections, DetectionClass,'_',
                SoundFilePathShort[x],
                'GibbonAlexNetAllFilesMalaysia.txt',
                sep = '')

        write.table(
          x = RavenSelectionTableDFAlexNet,
          sep = "\t",
          file = csv.file.name,
          row.names = FALSE,
          quote = FALSE
        )
        print(paste(
          "Saving Selection Table"
        ))
      }


    }
  }

  if(nrow(RavenSelectionTableDFAlexNet) == 0){
    csv.file.name <-
      paste(OutputFolderSelections, DetectionClass,'_',
            SoundFilePathShort[x],
            'GibbonAlexNetAllFilesMalaysia.txt',
            sep = '')


    ColNames <-  c(
      "Selection",
      "View",
      "Channel",
      "Begin Time (s)",
      "End Time (s)",
      "Low Freq (Hz)",
      "High Freq (Hz)",
      'Probability',
      "Detections"
    )

    TempNARow <- t(as.data.frame(rep(NA,length(ColNames))))

    colnames(TempNARow) <- ColNames

    write.table(
      x = TempNARow,
      sep = "\t",
      file = csv.file.name,
      row.names = FALSE,
      quote = FALSE
    )
    print(paste(
      "Saving Selection Table"
    ))
  }

  rm(TempWav)
  rm(short.sound.files)
  rm( test_ds )
  rm(short.wav)
  end.time.detection <- Sys.time()
  print(end.time.detection-start.time.detection)
  gc()
}, error = function(e) { cat("ERROR :", conditionMessage(e), "\n") })
}



