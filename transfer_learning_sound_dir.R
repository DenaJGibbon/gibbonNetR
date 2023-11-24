#' Transfer Learning from Sound Directories
#'
#' This function processes sound data from a specified directory, performs transfer learning using a pre-trained deep learning model, and saves the results.
#'
#' @param output_folder A character string specifying the path to the output folder where the results will be saved.
#' @param output_folder_selections A character string specifying the path to the folder where selection tables will be saved.
#' @param output_folder_wav A character string specifying the path to the folder where extracted WAV files will be saved.
#' @param top_model_path A character string specifying the path to the pre-trained top model for classification.
#' @param path_to_files A character string specifying the path to the directory containing sound files to process.
#' @param clip_duration The duration of each sound clip in seconds.
#' @param hop_size The hop size for splitting the sound clips.
#' @param downsample_rate The downsample rate for audio in Hz, set to 'NA' if no downsampling is required.
#' @param threshold The threshold for audio detection.
#' @param save_wav A logical value indicating whether to save the extracted sound clips as WAV files.
#' @param unique_classes_training A character vector containing the unique classes for training the model.
#' @param noise_category A character string specifying the noise category for exclusion.
#' @param max_freq_khz The maximum frequency in kHz for spectrogram visualization.
#' @param single_class A logical value indicating whether to process only a single class.
#' @param single_class_category A character string specifying the single class category when 'single_class' is set to TRUE.
#'
#' @details This function processes sound data from a directory, extracts sound clips, converts them to images, performs image classification using a pre-trained deep learning model, and saves the results including selection tables and image and audio files.
#'
#' @examples
#' \dontrun{
#' transfer_learning_sound_dir(
#'   output_folder = "output_results",
#'   output_folder_selections = "selection_tables",
#'   output_folder_wav = "extracted_audio",
#'   top_model_path = "pretrained_model.pth",
#'   path_to_files = "sound_data_directory"
#' )
#' }
#'


transfer_learning_sound_dir <- function(
    output_folder,
    output_folder_selections,
    output_folder_wav,
    top_model_path,
    path_to_files,
    clip_duration = 12,
    hop_size = 6,
    downsample_rate = 16000,
    threshold = 0.5,
    save_wav = TRUE,
    unique_classes_training = c('duet','hornbill.helmeted','hornbill.rhino','long.argus','noise'),
    noise_category = 'noise',
    max_freq_khz = 2,
    single_class = TRUE,
    single_class_category = 'hornbill.helmeted'
) {


  # Create output folders if they don't exist
  dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)
  dir.create(output_folder_selections, recursive = TRUE, showWarnings = FALSE)
  dir.create(output_folder_wav, recursive = TRUE, showWarnings = FALSE)

  for(x in (1:length(SoundFilePathFull)) ){ tryCatch({
    RavenSelectionTableDFTopModel <- data.frame()

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
      unlink(paste( tempdir(), '/WavFiles', sep=''), recursive = TRUE)
      unlink(paste( tempdir(), '/Images/Images', sep=''), recursive = TRUE)

      dir.create(paste( tempdir(), '/WavFiles', sep=''))
      dir.create(paste( tempdir(), '/Images/Images', sep=''), recursive = TRUE)

      WavFileTempDir <- paste( tempdir(), '/WavFiles', sep='')
      ImageTempDir <- paste( tempdir(), '/Images/Images', sep='')

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
        writeWave(short.sound.files[[d]],paste(WavFileTempDir,'/',
                                               SoundFilePathShort[x],'_',start.time[d], '.wav', sep=''),
                  extensible = F)
      }

      # Save images to a temp folder
      print(paste('Creating images',start.time[1],'start time clips'))

      for(e in 1:length(short.sound.files)){
        jpeg(paste(ImageTempDir,'/', SoundFilePathShort[x],'_',start.time[e],'.jpg',sep=''),res = 50)
        short.wav <- short.sound.files[[e]]

        seewave::spectro(short.wav,tlab='',flab='',axisX=F,axisY = F,scale=F,grid=F,flim=c(0.4,MaxFreq.khz),fastdisp=TRUE,noisereduction=1)

        graphics.off()
      }

      # Predict using TopModel ----------------------------------------------------
      print('Classifying images using Top Model')

      test.input <- ImageTempDir

      # ResNet
      test_ds <- image_folder_dataset(
        file.path(test.input),
        transform = . %>%
          torchvision::transform_to_tensor() %>%
          torchvision::transform_resize(size = c(224, 224)) %>%
          torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)))

      # TopModel and VGG19
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

      Detections <-  which(outputTableTopModel$Probability >= threshold &
                             outputTableTopModel$PredictedClass != noise.category )

      if(single.class =='TRUE'){
        Detections <-  which(outputTableTopModel$Probability >= threshold &
                               outputTableTopModel$PredictedClass == single.class.category )


      }

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
                          round(predicted_class_probability[DetectionIndices],2),
                          '_TopModel_.jpg', sep=''))

      if(sav.wav ==T){
        wav.file.paths <- list.files(WavFileTempDir,full.names = T)
        file.copy(wav.file.paths[DetectionIndices],
                  to= paste(OutputFolderWav,  DetectionClass,'_',
                            image.files.short[DetectionIndices],
                            '_',
                            round(predicted_class_probability[DetectionIndices],2),
                            '_TopModel_.wav', sep=''))
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
        Probability <- round(predicted_class_probability[DetectionIndices],2)

        RavenSelectionTableDFTopModelTemp <-
          cbind.data.frame(Selection,
                           View,
                           Channel,
                           MinFreq,
                           MaxFreq,start.time.new,end.time.new,Probability,
                           Detections)

        RavenSelectionTableDFTopModelTemp <-
          RavenSelectionTableDFTopModelTemp[, c(
            "Selection",
            "View",
            "Channel",
            "start.time.new",
            "end.time.new",
            "MinFreq",
            "MaxFreq",
            'Probability',"Detections"
          )]

        colnames(RavenSelectionTableDFTopModelTemp) <-
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

        RavenSelectionTableDFTopModel <- rbind.data.frame(RavenSelectionTableDFTopModel,
                                                         RavenSelectionTableDFTopModelTemp)

        if(nrow(RavenSelectionTableDFTopModel) > 0){
          csv.file.name <-
            paste(OutputFolderSelections, paste(unique(DetectionClass),'_',sep='-'),'_',
                  SoundFilePathShort[x],
                  'GibbonTopModelAllFilesMalaysia.txt',
                  sep = '')

          RavenSelectionTableDFTopModel$Class <- DetectionClass

          write.table(
            x = RavenSelectionTableDFTopModel,
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

    if(nrow(RavenSelectionTableDFTopModel) == 0){
      csv.file.name <-
        paste(OutputFolderSelections, paste(unique(DetectionClass),'_',sep='-'),'_',
              SoundFilePathShort[x],
              'GibbonTopModelAllFilesMalaysia.txt',
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

}


# example
transfer_learning_sound_dir(
  output_folder = "output_results",
  output_folder_selections = "selection_tables",
  output_folder_wav = "extracted_audio",
  top_model_path = "pretrained_model.pth",
  path_to_files = "sound_data_directory"
)
