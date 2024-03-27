#' Process and Save Spectrogram Images from Sound Files
#'
#' @param trainingBasePath Base directory containing the training folders.
#' @param outputBasePath Directory where the processed images will be saved.
#' @param splits Numeric vector specifying the split ratios for train, valid, and test sets. Defaults to c(0.8, 0.1, 0.1).
#' @param minfreq.khz Minimum frequency in kHz for the spectrogram. Defaults to 0.4.
#' @param maxfreq.khz Maximum frequency in kHz for the spectrogram. Defaults to 2.
#' @param new.sampleratehz New sample rate in Hz for resampling the audio. Defaults to 16000. Set to 'NA' if no resampling is required.
#'
#' @return Invisible NULL
#'
#' @examples
#' \dontrun{
#' spectrogram_images(
#'   trainingBasePath = "data/Clips",
#'   outputBasePath = "data/TrainingImages",
#'   splits = c(0.7, 0.2, 0.1)
#' )
#'}
#' @importFrom tuneR readWave
#' @importFrom seewave spectro
#' @importFrom tools file_path_sans_ext
#' @export

spectrogram_images <- function(trainingBasePath,
                               outputBasePath, splits,
                               minfreq.khz= 0.4, maxfreq.khz=1.6,new.sampleratehz=16000 ) {

  # Check if splits are valid
  if (sum(splits) != 1) stop("The sum of the splits must equal 1.")
  if (length(splits) != 3) stop("Exactly three split ratios should be provided.")

  # Lists all training folders
  TrainingFolders <- list.files(trainingBasePath, full.names = TRUE)
  TrainingFoldersShort <- list.files(trainingBasePath, full.names = FALSE)

  FolderVec <- c('train', 'valid', 'test') # Potential folders for classification

  for (z in seq_along(TrainingFolders)) {
    SoundFiles <- list.files(TrainingFolders[z], recursive = TRUE, full.names = TRUE)
    SoundFilesShort <- list.files(TrainingFolders[z], recursive = TRUE, full.names = FALSE)
    total_files <- length(SoundFiles)

    # Calculate indices for splitting
    # Shuffle indices
    shuffled_indices <- sample(1:total_files)

    # Calculate indices for splitting
    train_n <- floor(splits[1] * total_files)
    valid_n <- floor(splits[2] * total_files)

    train_idx <- shuffled_indices[1:train_n]
    valid_idx <- shuffled_indices[(train_n + 1):(train_n + valid_n)]
    test_idx <- shuffled_indices[(train_n + valid_n + 1):total_files]

    if(splits[1]==0){
      train_idx <- 0
    }

    if(splits[2]==0){
      valid_idx <-0
    }

    if(splits[3]==0){
      test_idx <- 0
    }

    for (y in seq_along(SoundFiles)) {
      # Determine the DataType based on the index
      if (y %in% train_idx) {
        DataType <- FolderVec[1]
      } else if (y %in% valid_idx) {
        DataType <- FolderVec[2]
      } else {
        DataType <- FolderVec[3]
      }

      subset_directory <- file.path(outputBasePath, DataType, TrainingFoldersShort[z])

      if (!dir.exists(subset_directory)) {
        dir.create(subset_directory, recursive = TRUE)
        message('Created output dir: ', subset_directory)
      } else {
        message(subset_directory, ' already exists')
      }

      wav_rm <- tools::file_path_sans_ext(SoundFilesShort[y])
      jpeg_filename <- file.path(subset_directory, paste0(wav_rm, '.jpg'))

      jpeg(jpeg_filename, res = 50)

      short_wav <- tuneR::readWave(SoundFiles[y])

      if(new.sampleratehz != 'NA'){
      short_wav <- tuneR::downsample(short_wav,new.sampleratehz)
      }

      seewave::spectro(short_wav, tlab = '', flab = '', axisX = FALSE,
                       axisY = FALSE, scale = FALSE, flim = c(minfreq.khz, maxfreq.khz), grid = FALSE)
      graphics.off()
    }
  }

  invisible(NULL)
}
