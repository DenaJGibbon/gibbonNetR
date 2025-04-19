test_that("Outputs appropriate test file", {
  # Set file path
  filepath <- system.file("extdata", "binary/", package = "gibbonNetR")

  # Train simple CNN model
  train_CNN_binary(
    input.data.path = filepath,
    test.data = paste(filepath, "/test/", sep = ""),
    architecture = "alexnet", # Choose 'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', or 'resnet152'
    unfreeze.param = TRUE,
    batch_size = 6,
    learning_rate = 0.001,
    epoch.iterations = 1, # Or any other list of integer epochs
    early.stop = "yes",
    save.model = TRUE,
    output.base.path = paste(tempdir(), "/BinaryDir/", sep = ""),
    trainingfolder = "test_binary"
  )

  ListOutputFiles <- list.files(paste(tempdir(), "/BinaryDir/", sep = ""), recursive = T, full.names = T)
  TestOutPut <- ListOutputFiles[which(str_detect(ListOutputFiles, "performance_tables/"))]
  TempCSV <- read.csv(TestOutPut)

  expect_true(ncol(TempCSV) == 20)

  # Check that the data frame has expected columns
  expect_true(all(c("AUC", "Class", "TestDataPath") %in% colnames(TempCSV)))

  # Check that a model file was saved
  model_files <- ListOutputFiles[str_detect(ListOutputFiles, "\\.pt$")]
  expect_true(length(model_files) >= 1)

  # Confirm architecture used in filename
  expect_true(any(str_detect(basename(model_files), "alexnet")))

  # Check that accuracy/AUC is within valid range
  expect_true(all(TempCSV$AUC >= 0 & TempCSV$AUC <= 1))

  # Confirm output subfolders created
  expect_true(any(str_detect(ListOutputFiles, "performance_tables/")))

})
