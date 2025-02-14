test_that("deploy_CNN_multi returns expected objects", {
  data("TempBinWav")

  # Save in temp directory
  dir.create(paste(tempdir(), "/MultiDir/Wav/", sep = ""), recursive = T, showWarnings = FALSE)

  # Write to temp directory
  writeWave(TempBinWav, filename = paste(tempdir(), "/MultiDir/Wav/", "TempBinWav.wav", sep = ""))

  # Find model path
  # Set model directory
  trained_models_dir <- system.file("extdata", "trainedresnetmulti/", package = "gibbonNetR")

  # Specify model path
  ModelPath <- list.files(trained_models_dir, full.names = T)

  # Deploy trained model over sound files
  deploy_CNN_multi(
    clip_duration = 12,
    architecture = "resnet18",
    output_folder = paste(tempdir(), "/MultiDir/Results/Images/", sep = ""),
    output_folder_selections = paste(tempdir(), "/MultiDir/Results/Selections/", sep = ""),
    output_folder_wav = paste(tempdir(), "/MultiDir/Results/Wavs/", sep = ""),
    detect_pattern = NA,
    top_model_path = ModelPath,
    path_to_files = paste(tempdir(), "/MultiDir/Wav/", sep = ""),
    downsample_rate = "NA",
    save_wav = F,
    class_names = c("female.gibbon", "hornbill.helmeted", "hornbill.rhino", "long.argus", "noise"),
    noise_category = "noise",
    single_class = FALSE,
    single_class_category = "female.gibbon",
    threshold = .25,
    max_freq_khz = 2
  )


  ListSelections <- list.files(paste(tempdir(), "/MultiDir/Results/Selections/", sep = ""), full.names = T)

  results <- ListSelections[1]

  expect_true(ncol(read.delim(results)) == 10)

  # Check that output file is created
  expect_true(file.exists(results))

  # Check that the data frame has expected columns
  expect_true(all(c("Probability", "Detections", "Class") %in% colnames(read.delim(results))))
})
