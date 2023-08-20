devtools::document()
devtools::load_all()


# Get setup for training --------------------------------------------------
setwd("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")
devtools::load_all()
# Location of spectrogram images for training
input.data.path <-  'data/imagesmalaysia/'

# Location of spectrogram images for testing
test.data.path <- 'data/imagesmalaysia/'

# Training data folder short
trainingfolder.short <- 'imagesmalaysia'

# Whether to unfreeze the layers
unfreeze.param <- TRUE # FALSE means the features are frozen; TRUE unfrozen

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Location to save the out
output.data.path <-paste('data/','output','unfrozen',unfreeze.param,trainingfolder.short,'/', sep='_')

# Create if doesn't exist
dir.create(output.data.path)

# Allow early stopping?
early.stop <- 'yes' # NOTE: Must comment out if don't want early stopping

gibbonNetR::train_alexnet(input.data.path=input.data.path,
                          test.data=test.data.path,
                          unfreeze = TRUE,
                          epoch.iterations=epoch.iterations,
                          early.stop = "yes",
                          output.base.path = "data/",
                          trainingfolder=trainingfolder.short,
                          positive.class="Gibbons",
                          negative.class="Noise")


gibbonNetR::train_VGG16(input.data.path=input.data.path,
                          test.data=test.data.path,
                          unfreeze = TRUE,
                          epoch.iterations=epoch.iterations,
                          early.stop = "yes",
                          output.base.path = "data/",
                          trainingfolder=trainingfolder.short,
                          positive.class="Gibbons",
                          negative.class="Noise")

gibbonNetR::train_VGG19(input.data.path=input.data.path,
                        test.data=test.data.path,
                        unfreeze = TRUE,
                        epoch.iterations=epoch.iterations,
                        early.stop = "yes",
                        output.base.path = "data/",
                        trainingfolder=trainingfolder.short,
                        positive.class="Gibbons",
                        negative.class="Noise")

gibbonNetR::train_ResNet18(input.data.path=input.data.path,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            early.stop = "yes",
                            output.base.path = "data/",
                            trainingfolder=trainingfolder.short,
                            positive.class="Gibbons",
                            negative.class="Noise")

gibbonNetR::train_ResNet50(input.data.path=input.data.path,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            early.stop = "yes",
                            output.base.path = "data/",
                            trainingfolder=trainingfolder.short,
                            positive.class="Gibbons",
                            negative.class="Noise")

gibbonNetR::train_ResNet152(input.data.path=input.data.path,
                            test.data=test.data.path,
                            unfreeze = TRUE,
                            epoch.iterations=epoch.iterations,
                            early.stop = "yes",
                            output.base.path = "data/",
                            trainingfolder=trainingfolder.short,
                            positive.class="Gibbons",
                            negative.class="Noise")

performancetables.dir <- '/Users/denaclink/Desktop/RStudioProjects/gibbonNetR/data/_output_unfrozen_TRUE_imagesmalaysia_/performance_tables/'
PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir)

PerformanceOutput$f1_plot
PerformanceOutput$pr_plot

