 #----eval=FALSE--------------------------------------------------------
 # Link to training clips on Zenodo
 ZenodoLink <- 'https://zenodo.org/records/14213067/files/trainingclips.zip?download=1'

 # Create data folder in your working directory
 dir.create("data")

 # Download into specified zip file location
 download.file(url = ZenodoLink, destfile = 'data/trainingclips.zip',method='curl')

 # Unzip folder
 exdir <- 'data/'
 utils::unzip(zipfile = 'data/trainingclips.zip', exdir = exdir )

 # Check folder composition
 TrainingDatapath <- paste(exdir,"trainingclips",sep='')

 # Check folder names
 list.files(TrainingDatapath)

 # Create spectrogram images
 spectrogram_images(
    trainingBasePath = TrainingDatapath,
    outputBasePath = 'data/trainingimages/',
    minfreq.khz = 0.4,
    maxfreq.khz = 1.6,
    random=FALSE,
    splits = c(0.7, 0.3, 0), # Assign proportion to training, validation, or test folders
    new.sampleratehz = 'NA'
  )




 #----eval = FALSE------------------------------------------------------
 library(gibbonNetR)

 # Link to training clips on Zenodo
 ZenodoLink <- 'https://zenodo.org/records/14213067/files/testclips.zip?download=1'

 # Download into specified zip file location
 download.file(url = ZenodoLink, destfile = 'data/testclips.zip',method='curl')

 # Unzip folder
 exdir <- 'data/'
 utils::unzip(zipfile = 'data/testclips.zip', exdir = exdir )

 # Check folder composition
 TestDatapath <- paste(exdir,"testclips",sep='')

 # Check folder names
 list.files(TestDatapath)

 # Create spectorgram images
 spectrogram_images(
    trainingBasePath = TestDatapath,
    outputBasePath = 'data/testimages/',
    minfreq.khz = 0.4,
    maxfreq.khz = 1.6,
    splits = c(0, 0, 1), # Assign proportion to training, validation, or test folders
    new.sampleratehz = 'NA'
  )



 #----eval=FALSE--------------------------------------------------------
 # Link to training clips on Zenodo
 ZenodoLink <- 'https://zenodo.org/records/14213067/files/trainingclipsbinary.zip?download=1'

 # Create data folder in your working directory
 dir.create("data")

 # Download into specified zip file location
 download.file(url = ZenodoLink, destfile = 'data/trainingclipsbinary.zip',method='curl')

 # Unzip folder
 exdir <- 'data/'
 utils::unzip(zipfile = 'data/trainingclipsbinary.zip', exdir = exdir )

 # Check folder composition
 TrainingDatapath <- paste(exdir,"trainingclipsbinary",sep='')

 # Check folder names
 list.files(TrainingDatapath)

 # Create spectrogram images
 spectrogram_images(
    trainingBasePath = TrainingDatapath,
    outputBasePath = 'data/trainingimagesbinary/',
    minfreq.khz = 0.4,
    maxfreq.khz = 1.6,
    random=FALSE,
    splits = c(0.7, 0.3, 0), # Assign proportion to training, validation, or test folders
    new.sampleratehz = 'NA'
  )




 ----eval = FALSE------------------------------------------------------
 library(gibbonNetR)

 # Link to training clips on Zenodo
 ZenodoLink <- 'https://zenodo.org/records/14213067/files/testclipsbinary.zip?download=1'

 # Download into specified zip file location
 download.file(url = ZenodoLink, destfile = 'data/testclipsbinary.zip',method='curl')

 # Unzip folder
 exdir <- 'data/'
 utils::unzip(zipfile = 'data/testclipsbinary.zip', exdir = exdir )

 # Check folder composition
 TestDatapath <- paste(exdir,"testclipsbinary",sep='')

 # Check folder names
 list.files(TestDatapath)

 # Create spectorgram images
 spectrogram_images(
    trainingBasePath = TestDatapath,
    outputBasePath = 'data/testimagesbinary/',
    minfreq.khz = 0.4,
    maxfreq.khz = 1.6,
    splits = c(0, 0, 1), # Assign proportion to training, validation, or test folders
    new.sampleratehz = 'NA'
  )


