# .onLoad <- function(libname, pkgname) {
#   if (!requireNamespace("torch", quietly = TRUE)) {
#     stop("Package 'torch' (>= 0.13.0) is required but not installed.")
#   }
#   if (packageVersion("torch") != "0.13.0") {
#     stop("Incorrect 'torch' version detected. Please install torch 0.13.0 using remotes::install_version('torch', version = '0.13.0').")
#   }
# }
