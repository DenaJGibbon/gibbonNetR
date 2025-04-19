test_that("train_CNN_multi works", {
  input.data.path <- system.file("extdata", "multiclass/", package = "gibbonNetR")

  test.data <- system.file("extdata", "multiclass/test/", package = "gibbonNetR")

  train_CNN_multi(
    input.data.path = input.data.path,
    test.data = test.data,
    architecture = "resnet18", # Choose architecture
    unfreeze.param = TRUE,
    class_weights = rep((1 / 5), 5),
    batch_size = 6,
    learning_rate = 0.001,
    epoch.iterations = 1, # Or any other list of integer epochs
    early.stop = "yes",
    output.base.path = paste(tempdir(), "/", sep = ""),
    trainingfolder = "test",
    noise.category = "noise",
    save.model = TRUE
  )


  ListOutputFiles <- list.files(paste(tempdir(), "/", sep = ""), recursive = T, full.names = T)
  TestOutPut <- ListOutputFiles[which(str_detect(ListOutputFiles, "performance_tables_multi/"))]
  TempCSV <- read.csv(TestOutPut[1])

  expect_true(ncol(TempCSV) == 20)

  # Check that the data frame has expected columns
  expect_true(all(c("AUC", "Class", "TestDataPath") %in% colnames(TempCSV)))

  # Check that a model file was saved
  model_files <- ListOutputFiles[str_detect(ListOutputFiles, ".pt$")]
  expect_true(length(model_files) >= 1)

  # Check model file naming pattern
  expect_true(any(str_detect(basename(model_files), "resnet18")))

  # Check that output folder structure is created correctly
  expect_true(any(str_detect(ListOutputFiles, "performance_tables_multi/")))

  # Check that accuracy or AUC values are numeric and in expected range
  expect_true(all(TempCSV$AUC >= 0 & TempCSV$AUC <= 1))

  # Check for expected levels in the Class column
  expect_true("female.gibbon" %in% unique(TempCSV$Class))
})
