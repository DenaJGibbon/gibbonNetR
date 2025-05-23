test_that("Outputs expected dataframe", {
  trained_models_dir <- system.file("extdata", "trainedresnetmulti/", package = "gibbonNetR")

  image_data_dir <- system.file("extdata", "multiclass/test/", package = "gibbonNetR")

  evaluate_trainedmodel_performance_multi(
    trained_models_dir = trained_models_dir,
    image_data_dir = image_data_dir,
    class_names = c("female.gibbon", "hornbill.helmeted", "hornbill.rhino", "long.argus", "noise"),
    output_dir = paste(tempdir(), "/data/", sep = ""), # Output directory for evaluation results
    noise.category = "noise"
  ) # Label for negative class

  # Find the location of saved evaluation files
  CSVName <- list.files(paste(tempdir(), "/data/performance_tables_multi_trained/", sep = ""), recursive = TRUE, full.names = TRUE)

  # Read in results
  results <- read.csv(CSVName[1])

  # Check the output of the first file
  expect_true(ncol(results) == 19)

  # Check that the data frame has expected columns
  expect_true(all(c("CNN.Architecture", "AUC", "Class") %in% colnames(as.data.frame(results))))

  # Check some useful metrics are there
  expect_true(all(c("Precision", "Recall", "F1", "Specificity", "Sensitivity") %in% colnames(results)))

  # Ensure all provided classes are represented
  expect_true(all(c("female.gibbon", "hornbill.rhino") %in% results$Class))
})
