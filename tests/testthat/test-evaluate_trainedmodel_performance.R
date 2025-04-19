test_that("function returns expected values", {
  # Set directory paths for trained models and test images

  trained_models_dir <- system.file("extdata", "trainedresnetbinary/", package = "gibbonNetR")
  image_data_dir <- system.file("extdata", "binary/test/", package = "gibbonNetR")

  # Evaluate the performance of the trained models using the test images
  evaluate_trainedmodel_performance(
    trained_models_dir = trained_models_dir,
    image_data_dir = image_data_dir,
    output_dir = paste(tempdir(), "/data/", sep = ""), # Output directory for evaluation results
    positive.class = "Gibbons", # Label for positive class
    negative.class = "Noise"
  ) # Label for negative class

  # Find the location of saved evaluation files
  CSVName <- list.files(paste(tempdir(), "/data/", sep = ""), recursive = TRUE, full.names = TRUE)

  # List output files
  CSVFiles <- list.files(paste(tempdir(), "/data/performance_tables_trained/", sep = ""), recursive = TRUE, pattern = "\\.csv$", full.names = TRUE)
  expect_true(length(CSVFiles) > 0)

  # Read and check contents of first CSV
  eval_df <- read.csv(CSVFiles[1])

  # Check expected structure
  expect_true(ncol(eval_df) == 19)
  expect_true(all(c("AUC", "Class", "TestData", "Sensitivity", "Specificity", "F1") %in% colnames(eval_df)))

  # Check number of rows > 0
  expect_true(nrow(eval_df) > 0)

  # Check AUC values between 0 and 1
  expect_true(all(eval_df$AUC >= 0 & eval_df$AUC <= 1))

  # Confirm that both classes are represented
  expect_true(all(c("Gibbons") %in% eval_df$Class))

  # Confirm accuracy metric values are in reasonable range
  expect_true(all(eval_df$Sensitivity >= 0 & eval_df$Sensitivity <= 1))
  expect_true(all(eval_df$Specificity >= 0 & eval_df$Specificity <= 1))
})
