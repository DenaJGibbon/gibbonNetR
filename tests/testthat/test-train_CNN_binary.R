test_that("Outputs appropriate test file", {

  # Set file path
  filepath <- system.file("extdata", "binary/", package = "gibbonNetR")

  # Train simple CNN model
  train_CNN_binary(
    input.data.path = filepath,
    test.data = paste(filepath,'/test/',sep=''),
    architecture = "alexnet",   #Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'
    unfreeze.param = TRUE,
    batch_size = 6,
    learning_rate = 0.001,
    epoch.iterations = 1,   # Or any other list of integer epochs
    early.stop = "yes",
    save.model= F,
    output.base.path = paste(tempdir(),'/BinaryDir/',sep=''),
    trainingfolder = "test_binary"
  )

  ListOutputFiles <- list.files(paste(tempdir(),'/BinaryDir/',sep=''),recursive = T,full.names = T)
  TestOutPut <- ListOutputFiles[which(str_detect(ListOutputFiles,'performance_tables/'))]
  TempCSV <- read.csv(TestOutPut)
  expect_true(ncol(TempCSV)==19)
  })
