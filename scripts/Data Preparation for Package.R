library(jpeg)
#utils::download.file(url='https://zenodo.org/records/10790619/files/ZenodoData.zip?download=1', 'data/zenodo/zenodo.zip', mode = "wb")
#utils::unzip('data/zenodo/zenodo.zip', exdir ='data/zenodo/')


# Define the directory containing training images
TrainingImages <- list.files('/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/zenodo/',
                             recursive = TRUE, full.names = TRUE)

# Get a list of training images and their full paths
TrainingImagesShort <- list.files('/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/zenodo/',
                                  recursive = TRUE, full.names = FALSE)

# Combine image data and filenames into a list
CombinedImageFiles <- list()

for(a in 1:length(TrainingImages)){
  # Read each image
  TempImage <- jpeg::readJPEG(TrainingImages[a])
  # Get the corresponding filename
  TempName <- TrainingImagesShort[a]
  # Combine image and filename into a list and store in CombinedList
  CombinedImageFiles[[a]] <- list(TempImage,TempName)
}

# Save combined_list as an RDA file
saveRDS(CombinedImageFiles, file = "data/TrainingImagesMulti.rda")

usethis::use_data(CombinedImageFiles)

# Load the saved RDA file
TrainingImagesList <- readRDS(file = "data/TrainingImagesMulti.rda")

for(b in 1:length(CombinedList)){
  # Retrieve image and filename from CombinedList
  TempImage <- CombinedList[[b]][[1]]
  # Create directory structure to save the image
  dir.create(paste(tempdir(), '/images/',  dirname(CombinedList[[b]][[2]]), sep=''),recursive = TRUE)
  # Define the path to save the image
  FilePath <- paste(tempdir(), '/images/', CombinedList[[b]][[2]], sep='')
  # Write the image to file
  jpeg::writeJPEG(image=TempImage,target=FilePath)
}

# List the saved images
list.files(paste(tempdir(), '/images/',sep=''))
