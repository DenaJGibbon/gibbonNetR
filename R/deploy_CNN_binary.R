#' Transfer Learning from Sound Directories
#'
#' This function processes sound data from a specified directory, performs transfer learning using a pre-trained deep learning model, and saves the results.
#'
#' @param output_folder A character string specifying the path to the output folder where the results will be saved.
#' @param output_folder_selections A character string specifying the path to the folder where selection tables will be saved.
#' @param output_folder_wav A character string specifying the path to the folder where extracted WAV files will be saved.
#' @param top_model_path A character string specifying the path to the pre-trained top model for classification.
#' @param path_to_files A character string specifying the path to the directory containing sound files to process.
#' @param detect_pattern (optional) A character string specifying a pattern to detect in the file names. Default is NA.
#' @param architecture A character string specifying the architecture of the pre-trained model.
#' @param clip_duration The duration of each sound clip in seconds.
#' @param hop_size The hop size for splitting the sound clips.
#' @param windowlength window length for input into 'spectro' function from seewave. Deafults to 512.
#' @param downsample_rate The downsample rate for audio in Hz. Set to 'NA' if no downsampling is required.
#' @param threshold The threshold for audio detection.
#' @param save_wav A logical value indicating whether to save the extracted sound clips as WAV files. Default is TRUE.
#' @param positive.class A character string specifying the positive class label. Default is 'Gibbons'.
#' @param negative.class A character string specifying the negative class label. Default is 'Noise'.
#' @param min_freq_khz The minimum frequency in kHz for spectrogram visualization.
#' @param max_freq_khz The maximum frequency in kHz for spectrogram visualization. Default is 2.
#' @return Returns spectrogram images, wav files (if specified), and Raven selection tables for each sound file in specified directory.
#' @details This function processes sound data from a directory, extracts sound clips, converts them to images, performs image classification using a pre-trained deep learning model, and saves the results including selection tables and image and audio files.
#' @note This function requires a model trained using the 'train_CNN_binary' function as input.
#' @examples
#' {
#'   #' Load data
#'   data("TempBinWav")
#'
#'   #' Create necessary directories
#'   dir.create(file.path(tempdir(), "BinaryDir", "Wav"), recursive = TRUE, showWarnings = TRUE)
#'
#'   #' Write to temp directory
#'   writeWave(TempBinWav, filename = file.path(tempdir(), "BinaryDir", "Wav", "TempBinWav.wav"))
#'
#'   #' Set model directory
#'   trained_models_dir <- system.file("extdata", "trainedresnetbinary", package = "gibbonNetR")
#'
#'   #' Specify model path
#'   ModelPath <- list.files(trained_models_dir, full.names = TRUE)
#'
#'   #' Run model
#'   deploy_CNN_binary(
#'     clip_duration = 12,
#'     architecture = "alexnet",
#'     output_folder = file.path(tempdir(), "BinaryDir", "Results", "Images"),
#'     output_folder_selections = file.path(tempdir(), "BinaryDir", "Results", "Selections"),
#'     output_folder_wav = file.path(tempdir(), "BinaryDir", "Results", "Wavs"),
#'     detect_pattern = NA,
#'     top_model_path = ModelPath,
#'     path_to_files = file.path(tempdir(), "BinaryDir", "Wav"),
#'     downsample_rate = "NA",
#'     threshold = 0.5,
#'     save_wav = FALSE,
#'     positive.class = "Gibbons",
#'     negative.class = "Noise",
#'     max_freq_khz = 2
#'   )
#' }
#' @importFrom grDevices jpeg graphics.off
#' @importFrom stats predict median
#' @importFrom magrittr %>%
#' @importFrom utils write.csv read.csv write.table
#' @importFrom tuneR readWave writeWave
#' @importFrom stringr str_detect str_count str_split_fixed
#' @importFrom seewave spectro cutw
#' @importFrom torch torch_tensor as_array torch_sigmoid
#' @importFrom torchvision transform_to_tensor transform_color_jitter transform_resize transform_center_crop transform_normalize
#' @importFrom luz luz_load
#' @importFrom tools file_path_sans_ext
#' @export

deploy_CNN_binary <- function(output_folder,
                              output_folder_selections,
                              output_folder_wav,
                              top_model_path,
                              path_to_files,
                              windowlength= 512,
                              detect_pattern = NA,
                              architecture,
                              clip_duration = 12,
                              hop_size = 6,
                              downsample_rate = 16000,
                              threshold = 0.5,
                              save_wav = TRUE,
                              positive.class = "Gibbons",
                              negative.class = "Noise",
                              min_freq_khz = 0.4,
                              max_freq_khz = 2) {
  # Create output folders if they don't exist
  dir.create(output_folder,
    recursive = TRUE,
    showWarnings = FALSE
  )
  dir.create(output_folder_selections,
    recursive = TRUE,
    showWarnings = FALSE
  )
  dir.create(output_folder_wav,
    recursive = TRUE,
    showWarnings = FALSE
  )


  if (is.list(path_to_files) == FALSE) {
    path_to_files <-
      list.files(path_to_files,
        recursive = TRUE,
        full.names = TRUE
      )
  }

  if (any(is.na(detect_pattern)) == FALSE) {
    path_to_files_long <- list()

    for (a in 1:length(detect_pattern)) {
      message(paste(
        "identifying sound files with the following pattern",
        detect_pattern[a]
      ))
      path_to_files_long[[a]] <-
        path_to_files[str_detect(path_to_files, c(detect_pattern[a]))]
    }

    path_to_files_long <- unlist(path_to_files_long)
  } else {
    path_to_files_long <- unlist(path_to_files)
  }

  path_to_files_short <- basename((path_to_files_long))
  TopModel <- luz_load(top_model_path)

  for (x in (1:length(path_to_files_long))) {
    tryCatch(
      {
        #
        RavenSelectionTableDFTopModel <- data.frame()

        start.time.detection <- Sys.time()
        message(paste(x, "out of", length(path_to_files_long)))
        message(path_to_files_short[x])
        TempWav <- readWave(path_to_files_long[x])
        WavDur <- duration(TempWav)

        Seq.start <- list()
        Seq.end <- list()

        i <- 1
        while (i + clip_duration < WavDur) {
          # message(i)
          Seq.start[[i]] <- i
          Seq.end[[i]] <- i + clip_duration
          i <- i + hop_size
        }


        ClipStart <- unlist(Seq.start)
        ClipEnd <- unlist(Seq.end)

        TempClips <- cbind.data.frame(ClipStart, ClipEnd)


        # Subset sound clips for classification -----------------------------------
        message("saving sound clips")
        set.seed(13)
        length <- nrow(TempClips)

        if (length > 100) {
          length.files <- seq(1, length, 100)
        } else {
          length.files <- c(1, length)
        }

        for (q in 1:(length(length.files) - 1)) {
          unlink(paste(tempdir(), "/WavFiles", sep = ""), recursive = TRUE)
          unlink(paste(tempdir(), "/Images/Images", sep = ""), recursive = TRUE)

          dir.create(paste(tempdir(), "/WavFiles", sep = ""))
          dir.create(paste(tempdir(), "/Images/Images", sep = ""), recursive = TRUE)

          WavFileTempDir <- paste(tempdir(), "/WavFiles", sep = "")
          ImageTempDir <- paste(tempdir(), "/Images/Images", sep = "")

          RandomSub <- seq(length.files[q], length.files[q + 1], 1)

          if (q == (length(length.files) - 1)) {
            RandomSub <- seq(length.files[q], length, 1)
          }

          start.time <- TempClips$ClipStart[RandomSub]
          end.time <- TempClips$ClipEnd[RandomSub]

          short.sound.files <- lapply(
            1:length(start.time),
            function(i) {
              extractWave(
                TempWav,
                from = start.time[i],
                to = end.time[i],
                xunit = c("time"),
                plot = F,
                output = "Wave"
              )
            }
          )

          if (downsample_rate != "NA") {
            message("downsampling")
            short.sound.files <- lapply(
              1:length(short.sound.files),
              function(i) {
                downsample(short.sound.files[[i]],
                  samp.rate = downsample_rate
                )
              }
            )
          }

          for (d in 1:length(short.sound.files)) {
            # message(d)
            writeWave(
              short.sound.files[[d]],
              paste(
                WavFileTempDir,
                "/",
                path_to_files_short[x],
                "_",
                start.time[d],
                ".wav",
                sep = ""
              ),
              extensible = F
            )
          }

          # Save images to a temp folder
          message(paste("Creating images", start.time[1], "start time clips"))

          for (e in 1:length(short.sound.files)) {
            jpeg(
              paste(
                ImageTempDir,
                "/",
                path_to_files_short[x],
                "_",
                start.time[e],
                ".jpg",
                sep = ""
              ),
              res = 50
            )
            short.wav <- short.sound.files[[e]]

            seewave::spectro(
              short.wav,
              tlab = "",
              flab = "",
              wl = windowlength,
              axisX = F,
              axisY = F,
              scale = F,
              grid = F,
              flim = c(min_freq_khz, max_freq_khz),
              fastdisp = TRUE,
              noisereduction = 1
            )

            graphics.off()
          }

          # Predict using TopModel ----------------------------------------------------
          message("Classifying images using Top Model")

          test.input <- paste(tempdir(), "/Images/", sep = "")

          # Define transforms based on model type
          if (str_detect(architecture, pattern = "resnet")) {
            transform_list <- function(x) {
              x %>%
                torchvision::transform_to_tensor() %>%
                torchvision::transform_color_jitter() %>%
                torchvision::transform_resize(256) %>%
                torchvision::transform_center_crop(224) %>%
                torchvision::transform_normalize(
                  mean = c(0.485, 0.456, 0.406),
                  std = c(0.229, 0.224, 0.225)
                )
            }
          } else {
            transform_list <- function(x) {
              x %>%
                torchvision::transform_to_tensor() %>%
                torchvision::transform_resize(size = c(224, 224)) %>%
                torchvision::transform_normalize(
                  mean = c(0.485, 0.456, 0.406),
                  std = c(0.229, 0.224, 0.225)
                )
            }
          }

          test_ds <-
            image_folder_dataset(test.input, transform = transform_list)
          test_dl <-
            dataloader(test_ds, batch_size = 32, shuffle = FALSE)

          # Predict using TrainedModel
          TrainedModelPred <- predict(TopModel, test_dl)
          TrainedModelProb <- torch_sigmoid(TrainedModelPred)
          TrainedModelProb <-
            as_array(torch_tensor(TrainedModelProb, device = "cpu"))
          TrainedModelClass <-
            ifelse((TrainedModelProb) < 0.5,
              positive.class,
              negative.class
            )

          # Add the results to output tables
          outputTableTrainedModel <-
            cbind.data.frame(TrainedModelClass, TrainedModelProb)

          colnames(outputTableTrainedModel) <-
            c("PredictedClass", "Probability")

          image.files <-
            list.files(file.path(test.input),
              recursive = TRUE,
              full.names = TRUE
            )
          nslash <- str_count(image.files, "/") + 1
          nslash <- nslash[1]
          image.files.short <-
            str_split_fixed(image.files, pattern = "/", n = nslash)[, nslash]
          image.files.short <-
            str_split_fixed(image.files.short, pattern = ".jpg", n = 2)[, 1]

          message("Saving output")

          Detections <-
            which(outputTableTrainedModel$Probability <= (1 - threshold))

          outputTableTrainedModel$Probability <-
            1 - outputTableTrainedModel$Probability

          Detections <- split(Detections, cumsum(c(1, diff(Detections)) != 1))

          for (i in 1:length(Detections)) {
            TempList <- Detections[[i]]
            if (length(TempList) == 1) {
              Detections[[i]] <- TempList[1]
            }
            if (length(TempList) == 2) {
              Detections[[i]] <- TempList[2]
            }
            if (length(TempList) > 2) {
              Detections[[i]] <- median(TempList)
            }
          }

          DetectionIndices <- unname(unlist(Detections))

          DetectionClass <- positive.class

          message("Saving output")
          file.copy(
            image.files[DetectionIndices],
            to = paste(
              output_folder,
              '/',
              round(outputTableTrainedModel$Probability[DetectionIndices], 2),
              "_",
              DetectionClass,
              "_",
              image.files.short[DetectionIndices],
              "_TopModel_.jpg",
              sep = ""
            )
          )

          if (save_wav == T) {
            wav.file.paths <- list.files(WavFileTempDir, full.names = TRUE)
            file.copy(
              wav.file.paths[DetectionIndices],
              to = paste(
                output_folder_wav,
                '/',
                round(outputTableTrainedModel$Probability[DetectionIndices], 2),
                "_",
                DetectionClass,
                "_",
                image.files.short[DetectionIndices],
                "_TopModel_.wav",
                sep = ""
              )
            )
          }

          Detections <- image.files.short[DetectionIndices]


          if (length(Detections) > 0) {
            Selection <- seq(1, length(Detections))
            View <- rep("Spectrogram 1", length(Detections))
            Channel <- rep(1, length(Detections))
            MinFreq <- rep(100, length(Detections))
            MaxFreq <- rep(max_freq_khz * 1000, length(Detections))
            ndash <- str_count(Detections[1], pattern = "_") + 1
            start.time.new <-
              as.numeric(str_split_fixed(Detections, pattern = "_", n = ndash)[, ndash])
            end.time.new <- start.time.new + clip_duration
            Probability <-
              round(outputTableTrainedModel$Probability[DetectionIndices], 2)

            RavenSelectionTableDFTopModelTemp <-
              cbind.data.frame(
                Selection,
                View,
                Channel,
                MinFreq,
                MaxFreq,
                start.time.new,
                end.time.new,
                Probability,
                Detections
              )

            RavenSelectionTableDFTopModelTemp <-
              RavenSelectionTableDFTopModelTemp[, c(
                "Selection",
                "View",
                "Channel",
                "start.time.new",
                "end.time.new",
                "MinFreq",
                "MaxFreq",
                "Probability",
                "Detections"
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
                "Probability",
                "Detections"
              )
            RavenSelectionTableDFTopModelTemp$Class <- DetectionClass

            RavenSelectionTableDFTopModel <-
              rbind.data.frame(
                RavenSelectionTableDFTopModel,
                RavenSelectionTableDFTopModelTemp
              )



            if (nrow(RavenSelectionTableDFTopModel) > 0) {
              csv.file.name <-
                paste(
                  output_folder_selections,
                  paste(unique(DetectionClass), "_", sep = "-"),
                  "_",
                  path_to_files_short[x],
                  "TopModelBinary.txt",
                  sep = ""
                )



              write.table(
                x = RavenSelectionTableDFTopModel,
                sep = "\t",
                file = csv.file.name,
                row.names = FALSE,
                quote = FALSE
              )
              message(paste("Saving Selection Table", csv.file.name))
            }
          }
        }

        if (nrow(RavenSelectionTableDFTopModel) == 0) {
          csv.file.name <-
            paste(
              output_folder_selections,
              paste(unique(DetectionClass), "_", sep = "-"),
              "_",
              path_to_files_short[x],
              "TopModelBinary.txt",
              sep = ""
            )


          ColNames <- c(
            "Selection",
            "View",
            "Channel",
            "Begin Time (s)",
            "End Time (s)",
            "Low Freq (Hz)",
            "High Freq (Hz)",
            "Probability",
            "Detections"
          )

          TempNARow <- t(as.data.frame(rep(NA, length(ColNames))))

          colnames(TempNARow) <- ColNames

          write.table(
            x = TempNARow,
            sep = "\t",
            file = csv.file.name,
            row.names = FALSE,
            quote = FALSE
          )
          message(paste("Saving Selection Table"))
        }

        rm(TempWav)
        rm(TempClips)
        rm(short.sound.files)
        rm(test_ds)
        rm(short.wav)
        end.time.detection <- Sys.time()
        message(end.time.detection - start.time.detection)
        gc()
      },
      error = function(e) {
        cat("ERROR :", conditionMessage(e), "\n")
      }
    )
  }
}
