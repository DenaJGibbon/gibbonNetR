test_that("outputs spectrogram images as expected", {

   # Load data
   data("TempBinWav")

   # Define the output directory
   output.dir <- paste(tempdir(), '/MultiDir/Noise/train/',sep='')

   # Create the output directory
   dir.create(output.dir, recursive = TRUE, showWarnings = FALSE)

   # Define the intervals for cutting the wave
   cutwave.list <- seq(1, 30, 5)

   # Extract subsamples from the waveform
   subsamps <- lapply(1:(length(cutwave.list) - 1),
                      function(i)
                        extractWave(
                          TempBinWav,
                          from = cutwave.list[i],
                          to = cutwave.list[i + 1],
                          xunit = c("time"),
                          plot = FALSE,
                          output = "Wave"
                        ))

   # Write the extracted subsamples to .wav files
   lapply(1:length(subsamps),
          function(i)
            writeWave(
              subsamps[[i]],
              filename = paste(
                output.dir,
                'temp_', i, '_', '.wav',
                sep = ''
              ),
              extensible = FALSE
            )
   )

   # List the files in the output directory
   list.files(output.dir)

   # Generate spectrogram images
   spectrogram_images(
     trainingBasePath =  paste(tempdir(), '/MultiDir/Noise/',sep=''),
     outputBasePath = paste(tempdir(), 'Spectro/', sep = ''),
     splits = c(0.7, 0.3, 0),
     new.sampleratehz = 'NA',
     random =FALSE
   )

   # List the images generated
   ListImages <- list.files(paste(tempdir(), 'Spectro/', sep = ''), recursive = TRUE)

   print(ListImages)

   expect_true( length(ListImages)==5 )
   expect_true( unique(str_detect(ListImages,'.jpg')) ==TRUE)

})
