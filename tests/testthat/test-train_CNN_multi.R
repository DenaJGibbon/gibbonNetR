test_that("train_CNN_multi works", {

  # Set file path
  filepath <- system.file("extdata", "multiclass/", package = "gibbonNetR")

  # Train simple CNN model
  train_CNN_multi(
    input.data.path = filepath,
    test.data = paste(filepath,'/test/',sep=''),
    architecture = "alexnet",  # Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'
    unfreeze.param = TRUE,
    batch_size = 6,
    class_weights = rep( (1/5), 5),
    learning_rate = 0.001,
    epoch.iterations = 1,  # Or any other list of integer epochs
    early.stop = "yes",
    save.model= TRUE,
    output.base.path = paste(tempdir(),'/MultiDir/',sep=''),
    trainingfolder = "test_multi",
    noise.category = 'noise'
  )

  ListOutputFiles <- list.files(paste(tempdir(),'/MultiDir/',sep=''),recursive = T,full.names = T)
  TestOutPut <- ListOutputFiles[which(str_detect(ListOutputFiles,'performance_tables_multi/'))]
  TempCSV <- read.csv(TestOutPut[1])
  expect_true(ncol(TempCSV)==19)

})
