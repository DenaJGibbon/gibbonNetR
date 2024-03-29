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


TempBinWav <- readWave('/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/inst/extdata/testfilebinary/S19_20180214_08003_clip_binary.wav')
saveRDS(TempBinWav, file = "data/TempBinWav.rda")


usethis::use_data(TempBinWav)

TempResNetModel <- luz_load('/Users/denaclink/Desktop/RStudioProjects/Gibbon-transfer-learning-multispecies/model_output/_imagesmalaysia_binary_unfrozen_TRUE_/_imagesmalaysia_5_resnet50_model.pt')
usethis::use_data(TempResNetModel)

data("TempResNetModel")

# Example usage:
 dir.create(paste(tempdir(),'/embedding_model/'),recursive = T)

 ModelPathDir <- paste(tempdir(),'/embedding_model/','TempResNetModel.pt',sep='')

 # Write to temp directory
 luz_save(TempResNetModel,path=ModelPathDir)


# Evaluate performance ----------------------------------------------------

 # Set directory paths for trained models and test images
 trained_models_dir <- '/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/inst/extdata/trainedresnetbinary/'
 image_data_dir <- "inst/extdata/binary/test/"

 # Evaluate the performance of the trained models using the test images
 evaluate_trainedmodel_performance(trained_models_dir = trained_models_dir,
                                   image_data_dir = image_data_dir,
                                   output_dir = paste(tempdir(), '/data/'),  # Output directory for evaluation results
                                   positive.class = 'Gibbons',  # Label for positive class
                                   negative.class = 'Noise')    # Label for negative class

 # Find the location of saved evaluation files
 CSVName <- list.files(paste(tempdir(), '/data/'), recursive = TRUE, full.names = TRUE)

 # Check the output of the first file
 head(read.csv(CSVName[1]))

 get_best_performance(performancetables.dir=paste(tempdir(), '/data/performance_tables_trained/'),
                      model.type='binary',class='Gibbons',Thresh.val=0)



# Evaluate model performance multi-class ----------------------------------
 train_CNN_multi(
   input.data.path = "inst/extdata/multiclass/",
   test.data = "inst/extdata/multiclass/test/",
   architecture = "resnet18",  # Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'
   unfreeze.param = TRUE,
   batch_size = 6,
   class_weights = rep( (1/5), 5),
   learning_rate = 0.001,
   epoch.iterations = 3,  # Or any other list of integer epochs
   early.stop = "yes",
   save.model= TRUE,
   output.base.path = paste(tempdir(),'/MultiDir/',sep=''),
   trainingfolder = "test_multi",
   noise.category = 'noise'
 )

 TempFileList <- list.files(paste(tempdir(),'/MultiDir/',sep=''),full.names = T,recursive = T)

 ModelPath <- TempFileList[which(str_detect(TempFileList,'model.pt'))]

 trained_models_dir <-  paste(tempdir(),'/MultiDir/',sep='')
 image_data_dir <- "inst/extdata/multiclass/test/"

 evaluate_trainedmodel_performance_multi(trained_models_dir = trained_models_dir,
                                   image_data_dir = image_data_dir,
                                   class_names=c('duet','hornbill.helmeted','hornbill.rhino','long.argus','noise'),
                                   output_dir = paste(tempdir(), '/data/',sep=''),  # Output directory for evaluation results
                                   noise.category='noise')    # Label for negative class

