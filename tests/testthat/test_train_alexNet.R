library(testthat)

test_that("train_alexNet runs without errors on default settings", {
  input_data_path <- "path_to_sample_input_data"
  test_data_path <- "path_to_sample_test_data"

  result <- train_alexNet(input_data_path, test_data_path)

  expect_s3_class(result, "list")
  expect_true(file.exists(paste0(result$Output_Path, "AlexNetmodel_metadata.csv")))
})
