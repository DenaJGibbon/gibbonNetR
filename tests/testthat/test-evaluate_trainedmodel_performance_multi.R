test_that("Outputs expected dataframe", {
  trained_models_dir <- system.file("extdata", "trainedresnetmulti/", package = "gibbonNetR")
  image_data_dir <- system.file("extdata", "multiclass/test/", package = "gibbonNetR")


  evaluate_trainedmodel_performance_multi(trained_models_dir = trained_models_dir,
                                          image_data_dir = image_data_dir,
                                          class_names=c('duet','hornbill.helmeted','hornbill.rhino','long.argus','noise'),
                                          output_dir = paste(tempdir(), '/data/',sep=''),  # Output directory for evaluation results
                                          noise.category='noise')    # Label for negative class

  # Find the location of saved evaluation files
  CSVName <- list.files(paste(tempdir(), '/data/'), recursive = TRUE, full.names = TRUE)

  # Check the output of the first file
  expect_true( ncol(read.csv(CSVName[1]))==19)

})
