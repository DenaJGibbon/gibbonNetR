test_that("Function outputs expected .csv file", {

    # Load data
   data("TempBinWav")

   # Save in temp directory
   dir.create(paste(tempdir(),'/BinaryDir/Wav/',sep=''),recursive = T, showWarnings = FALSE)

   #Write to temp directory
   writeWave(TempBinWav,filename = paste(tempdir(),'/BinaryDir/Wav/','TempBinWav.wav',sep=''))

   # Set model directory
   trained_models_dir <- system.file("extdata", "trainedresnetbinary/", package = "gibbonNetR")


   TempFileList <- list.files(trained_models_dir,full.names = T,recursive = T)
   ModelPath <- TempFileList[which(str_detect(TempFileList,'.pt'))]

   deploy_CNN_binary (
     clip_duration = 12,
     architecture='alexnet',
     output_folder = paste(tempdir(),'/BinaryDir/Results/Images/',sep=''),
     output_folder_selections = paste(tempdir(),'/BinaryDir/Results/Selections/',sep=''),
     output_folder_wav = paste(tempdir(),'/BinaryDir/Results/Wavs/',sep=''),
     detect_pattern=NA,
     top_model_path = ModelPath,
     path_to_files = paste(tempdir(),'/BinaryDir/Wav/',sep=''),
     downsample_rate = 'NA',
     threshold = 0.5,
     save_wav = F,
     positive.class = 'Gibbons',
     negative.class = 'Noise',
     max_freq_khz = 2
   )

  ListSelections <-  list.files(paste(tempdir(),'/BinaryDir/Results/Selections/',sep=''),full.names = T)

  results <- ListSelections[1]

  expect_true( ncol( read.delim(results)) ==10)

  # Check that output file is created
  expect_true(file.exists(results))

  # Check that the data frame has expected columns
  expect_true(all(c("Probability", "Detections", "Class") %in% colnames(read.delim(results))))


})
