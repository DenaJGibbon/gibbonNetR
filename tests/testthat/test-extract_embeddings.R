test_that("Extract embeddings returns expected objects", {

   # Specify trained model dir
   trained_models_dir <- system.file("extdata", "trainedresnetbinary/", package = "gibbonNetR")

   # Create list of files in temp directory
   TempFileList <- list.files(trained_models_dir,full.names = T,recursive = T)

   # Find model path
   ModelPath <- TempFileList[which(str_detect(TempFileList,'model.pt'))]

   # Specify model path
   ImageFile <- system.file("extdata", "multiclass/test/", package = "gibbonNetR")

   # Function to extract and plot embeddings
   result <- extract_embeddings(test_input=ImageFile,
                                model_path =ModelPath,
                                target_class = "duet",
                                unsupervised='TRUE'
   )

   expect_true( length(result)==3 )
})
