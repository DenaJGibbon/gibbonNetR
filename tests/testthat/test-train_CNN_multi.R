test_that("train_CNN_multi works", {

    input.data.path <- system.file("extdata", "multiclass/", package = "gibbonNetR")

    test.data <- system.file("extdata", "multiclass/test/", package = "gibbonNetR")

    train_CNN_multi(
     input.data.path = input.data.path,
     test.data = test.data,
     architecture = "alexnet",  # Choose architecture
     unfreeze.param = TRUE,
     class_weights = rep( (1/5), 5),
     batch_size = 6,
     learning_rate = 0.001,
     epoch.iterations = 1,  # Or any other list of integer epochs
     early.stop = "yes",
     output.base.path = paste(tempdir(),'/',sep=''),
     trainingfolder = "test",
     noise.category = 'noise'
   )


  ListOutputFiles <- list.files(paste(tempdir(),'/',sep=''),recursive = T,full.names = T)
  TestOutPut <- ListOutputFiles[which(str_detect(ListOutputFiles,'performance_tables_multi/'))]
  TempCSV <- read.csv(TestOutPut[1])
  head(TempCSV)
  expect_true(ncol(TempCSV)==20)

})
