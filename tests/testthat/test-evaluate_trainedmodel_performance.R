test_that("function returns expected values", {
   # Set directory paths for trained models and test images

   trained_models_dir <- system.file("extdata", "trainedresnetbinary/", package = "gibbonNetR")
   image_data_dir <- system.file("extdata", "binary/test/", package = "gibbonNetR")

   # Evaluate the performance of the trained models using the test images
   evaluate_trainedmodel_performance(trained_models_dir = trained_models_dir,
                                     image_data_dir = image_data_dir,
                                     output_dir = paste(tempdir(), '/data/', sep=''),  # Output directory for evaluation results
                                     positive.class = 'Gibbons',   #Label for positive class
                                     negative.class = 'Noise')     #Label for negative class

   # Find the location of saved evaluation files
   CSVName <- list.files(paste(tempdir(), '/data/',sep=''), recursive = TRUE, full.names = TRUE)

   # Check the output of the first file
   head(read.csv(CSVName[1]))
   expect_true( ncol(read.csv(CSVName[1]))==19)
})
