library(stringr)
library(tuneR)
library(seewave)
set.seed(13)


# Danum validation data prep --------------------------------------------------------
overlap_threshold <-1/3
clip.duration <- 12
hop.size <- 4

DanumBoxDrive <-'/Volumes/DJC Files/Clink et al Zenodo Data/ValidationSoundFiles'

# Prepare selection tables ------------------------------------------------

# List selection table full names
SelectionTables <-
  list.files('/Volumes/DJC Files/Clink et al Zenodo Data/AnnotatedFilesValidation/',pattern = '.txt',full.names = T)

# List selection table short names
SelectionTablesShort <-
  list.files('/Volumes/DJC Files/Clink et al Zenodo Data/AnnotatedFilesValidation',pattern = '.txt')

# Remove .txt
SelectionTablesID <- str_split_fixed(SelectionTablesShort,pattern = '.Table',n=2)[,1]

# Start with one file
SoundFilePathFull <- list.files(DanumBoxDrive,full.names = T,recursive = T)

nslash <- str_count(SoundFilePathFull[1],'/')+1

SoundFilePathShort <- str_split_fixed(SoundFilePathFull,
                                      pattern = '/',n=nslash)[,nslash]

SoundFilePathShort <- str_split_fixed(SoundFilePathShort,
                                      pattern = '.wav',n=2)[,1]

AnnotationsPathFull <- SelectionTables


AnnotationsPathShort <- SelectionTablesID

for( b in 1: length(AnnotationsPathFull)){tryCatch({
  print(b)
  WavIndex <- which(str_detect(SoundFilePathShort,AnnotationsPathShort[b]))

  TempAnnotations <- read.delim2(AnnotationsPathFull[b])
  print(unique(TempAnnotations$Call.type))
  TempAnnotationsGibbon <- subset(TempAnnotations,Call.type=='female.gibbon')
  if(nrow(TempAnnotationsGibbon) >0 ){
    TempWav <- readWave(SoundFilePathFull[WavIndex])
    WavDur <- duration(TempWav)

    Seq.start <- list()
    Seq.end <- list()

    i <- 1
    while (i + clip.duration < WavDur) {
      # print(i)
      Seq.start[[i]] = i
      Seq.end[[i]] = i+clip.duration
      i= i+hop.size
    }

    ClipStart <- unlist(Seq.start)
    ClipEnd <- unlist(Seq.end)

    ClipDataFrame <- cbind.data.frame(ClipStart,ClipEnd)

    TempClipsCombinedNoise <- data.frame()

    for(c in 1:nrow(TempAnnotationsGibbon)){

      TempRow <- TempAnnotationsGibbon[c,]

      StartTime <- as.numeric(TempRow$Begin.Time..s.)
      EndTime <- as.numeric(TempRow$End.Time..s.)

      # Compute the overlap threshold
      overlap_threshold <-1/3

      # Duration of each clip
      ClipDataFrame$Duration <- ClipDataFrame$ClipEnd - ClipDataFrame$ClipStart

      # Duration of the overlap for each clip with the window
      ClipDataFrame$OverlapDuration <- pmin(ClipDataFrame$ClipEnd, EndTime) - pmax(ClipDataFrame$ClipStart, StartTime)

      # Only select clips where overlap duration is >= 2/3 of the clip's duration
      TempClips <- ClipDataFrame[which(ClipDataFrame$OverlapDuration >= overlap_threshold * ClipDataFrame$Duration &
                                         ClipDataFrame$OverlapDuration > 0),]

      if(nrow(TempClips)>0){
        TempClipsNoise <- ClipDataFrame[-which(ClipDataFrame$OverlapDuration >= overlap_threshold * ClipDataFrame$Duration &
                                           ClipDataFrame$OverlapDuration > 0),]
      } else{
        TempClipsNoise <- ClipDataFrame

      }

      TempClipsCombinedNoise <- rbind.data.frame(TempClipsCombinedNoise,TempClipsNoise )

      TempClass <- 'Gibbons'
      TempClassNoise <- 'Noise'


      if(nrow(TempClips) >1){

        subset.directory <- paste('/Volumes/DJC Files/Clink et al Zenodo Data/ValidationClipsDanum/',TempClass,sep='')

        if (!dir.exists(subset.directory)){
          dir.create(subset.directory)
          print(paste('Created output dir',subset.directory))
        } else {
          print(paste(subset.directory,'already exists'))
        }
        short.sound.files <- lapply(1:nrow(TempClips),
                                    function(i)
                                      extractWave(
                                        TempWav,
                                        from = TempClips$ClipStart[i],
                                        to = TempClips$ClipEnd[i],
                                        xunit = c("time"),
                                        plot = F,
                                        output = "Wave"
                                      ))

        short.sound.files <- lapply(1:length(short.sound.files),
                                    function(i)
                                      downsample(
                                        short.sound.files[[i]],16000
                                      ))

        for(d in 1:length(short.sound.files)){

          writeWave(short.sound.files[[d]],paste(subset.directory,'/',
                                                 TempClass,'_',AnnotationsPathShort[b],'_',TempClips$ClipEnd[d], '.wav', sep=''),
                    extensible = F)
        }

      }
    }


    if(nrow(TempClipsCombinedNoise) >1 ){
      subset.directory <- paste('/Volumes/DJC Files/Clink et al Zenodo Data/ValidationClipsDanum/',TempClassNoise,sep='')

      if (!dir.exists(subset.directory)){
        dir.create(subset.directory)
        print(paste('Created output dir',subset.directory))
      } else {
        print(paste(subset.directory,'already exists'))
      }
      short.sound.files <- lapply(1:nrow(TempClipsCombinedNoise),
                                  function(i)
                                    extractWave(
                                      TempWav,
                                      from = TempClipsCombinedNoise$ClipStart[i],
                                      to = TempClipsCombinedNoise$ClipEnd[i],
                                      xunit = c("time"),
                                      plot = F,
                                      output = "Wave"
                                    ))

      short.sound.files <- lapply(1:length(short.sound.files),
                                  function(i)
                                    downsample(
                                      short.sound.files[[i]],16000
                                    ))

      # Randomly choose some noise clips
      RanSeq <- sample(1:length(short.sound.files),16,replace = F)
      for(d in RanSeq){

        writeWave(short.sound.files[[d]],paste(subset.directory,'/',
                                               TempClassNoise,'_',AnnotationsPathShort[b],'_',TempClipsCombinedNoise$ClipEnd[d], '.wav', sep=''),
                  extensible = F)
      }

    }
  }
}, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}

# Danum Test  data prep --------------------------------------------------------
DanumBoxDrive <-'/Volumes/DJC Files/Clink et al Zenodo Data/TestSoundFiles'

# Prepare selection tables ------------------------------------------------

# List selection table full names
SelectionTables <-
  list.files('/Volumes/DJC Files/Clink et al Zenodo Data/AnnotatedFilesTest/',pattern = '.txt',full.names = T)

# List selection table short names
SelectionTablesShort <-
  list.files('/Volumes/DJC Files/Clink et al Zenodo Data/AnnotatedFilesTest',pattern = '.txt')

# Remove .txt
SelectionTablesID <- str_split_fixed(SelectionTablesShort,pattern = '.Table',n=2)[,1]

# Start with one file
SoundFilePathFull <- list.files(DanumBoxDrive,full.names = T,recursive = T)

nslash <- str_count(SoundFilePathFull[1],'/')+1

SoundFilePathShort <- str_split_fixed(SoundFilePathFull,
                                      pattern = '/',n=nslash)[,nslash]

SoundFilePathShort <- str_split_fixed(SoundFilePathShort,
                                      pattern = '.wav',n=2)[,1]

AnnotationsPathFull <- SelectionTables


AnnotationsPathShort <- SelectionTablesID

for( b in 1: length(AnnotationsPathFull)){tryCatch({
  print(b)
  WavIndex <- which(str_detect(SoundFilePathShort,AnnotationsPathShort[b]))

  TempAnnotations <- read.delim2(AnnotationsPathFull[b])
  print(unique(TempAnnotations$Call.type))
  TempAnnotationsGibbon <- subset(TempAnnotations,Call.type=='female.gibbon')
  if(nrow(TempAnnotationsGibbon) >0 ){
    TempWav <- readWave(SoundFilePathFull[WavIndex])
    WavDur <- duration(TempWav)

    Seq.start <- list()
    Seq.end <- list()

    i <- 1
    while (i + clip.duration < WavDur) {
      # print(i)
      Seq.start[[i]] = i
      Seq.end[[i]] = i+clip.duration
      i= i+hop.size
    }

    ClipStart <- unlist(Seq.start)
    ClipEnd <- unlist(Seq.end)

    ClipDataFrame <- cbind.data.frame(ClipStart,ClipEnd)

    TempClipsCombinedNoise <- data.frame()

    for(c in 1:nrow(TempAnnotationsGibbon)){

      TempRow <- TempAnnotationsGibbon[c,]

      StartTime <- as.numeric(TempRow$Begin.Time..s.)
      EndTime <- as.numeric(TempRow$End.Time..s.)

      # Compute the overlap threshold
      overlap_threshold <-1/3

      # Duration of each clip
      ClipDataFrame$Duration <- ClipDataFrame$ClipEnd - ClipDataFrame$ClipStart

      # Duration of the overlap for each clip with the window
      ClipDataFrame$OverlapDuration <- pmin(ClipDataFrame$ClipEnd, EndTime) - pmax(ClipDataFrame$ClipStart, StartTime)

      # Only select clips where overlap duration is >= 2/3 of the clip's duration
      TempClips <- ClipDataFrame[which(ClipDataFrame$OverlapDuration >= overlap_threshold * ClipDataFrame$Duration &
                                         ClipDataFrame$OverlapDuration > 0),]

      if(nrow(TempClips)>0){
        TempClipsNoise <- ClipDataFrame[-which(ClipDataFrame$OverlapDuration >= overlap_threshold * ClipDataFrame$Duration &
                                           ClipDataFrame$OverlapDuration > 0),]
      } else{
        TempClipsNoise <- ClipDataFrame

      }

      TempClipsCombinedNoise <- rbind.data.frame(TempClipsCombinedNoise,TempClipsNoise )

      TempClass <- 'Gibbons'
      TempClassNoise <- 'Noise'


      if(nrow(TempClips) >1){

        subset.directory <- paste('/Volumes/DJC Files/Clink et al Zenodo Data/TestClipsDanum/',TempClass,sep='')

        if (!dir.exists(subset.directory)){
          dir.create(subset.directory)
          print(paste('Created output dir',subset.directory))
        } else {
          print(paste(subset.directory,'already exists'))
        }
        short.sound.files <- lapply(1:nrow(TempClips),
                                    function(i)
                                      extractWave(
                                        TempWav,
                                        from = TempClips$ClipStart[i],
                                        to = TempClips$ClipEnd[i],
                                        xunit = c("time"),
                                        plot = F,
                                        output = "Wave"
                                      ))

        short.sound.files <- lapply(1:length(short.sound.files),
                                    function(i)
                                      downsample(
                                        short.sound.files[[i]],16000
                                      ))

        for(d in 1:length(short.sound.files)){

          writeWave(short.sound.files[[d]],paste(subset.directory,'/',
                                                 TempClass,'_',AnnotationsPathShort[b],'_',TempClips$ClipEnd[d], '.wav', sep=''),
                    extensible = F)
        }

      }
    }


    if(nrow(TempClipsCombinedNoise) >1 ){
      subset.directory <- paste('/Volumes/DJC Files/Clink et al Zenodo Data/TestClipsDanum/',TempClassNoise,sep='')

      if (!dir.exists(subset.directory)){
        dir.create(subset.directory)
        print(paste('Created output dir',subset.directory))
      } else {
        print(paste(subset.directory,'already exists'))
      }
      short.sound.files <- lapply(1:nrow(TempClipsCombinedNoise),
                                  function(i)
                                    extractWave(
                                      TempWav,
                                      from = TempClipsCombinedNoise$ClipStart[i],
                                      to = TempClipsCombinedNoise$ClipEnd[i],
                                      xunit = c("time"),
                                      plot = F,
                                      output = "Wave"
                                    ))

      short.sound.files <- lapply(1:length(short.sound.files),
                                  function(i)
                                    downsample(
                                      short.sound.files[[i]],16000
                                    ))

      # Randomly choose some noise clips
      RanSeq <- sample(1:length(short.sound.files),16,replace = F)
      for(d in RanSeq){

        writeWave(short.sound.files[[d]],paste(subset.directory,'/',
                                               TempClassNoise,'_',AnnotationsPathShort[b],'_',TempClipsCombinedNoise$ClipEnd[d], '.wav', sep=''),
                  extensible = F)
      }

    }
  }
}, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}

# Maliau data prep --------------------------------------------------------
overlap_threshold <- 0.5
clip.duration <- 12
hop.size <- 4

MaliauBoxDrive <-'/Volumes/DJC Files/Clink et al Zenodo Data/MaliauAnnotations/'

# Prepare selection tables ------------------------------------------------

# List selection table full names
SelectionTables <-
  list.files('/Volumes/DJC Files/Clink et al Zenodo Data/MaliauAnnotations/MaliauAnnotations/',pattern = '.txt',full.names = T)

# List selection table short names
SelectionTablesShort <-
  list.files('/Volumes/DJC Files/Clink et al Zenodo Data/MaliauAnnotations/MaliauAnnotations/',pattern = '.txt')

# Remove .txt
SelectionTablesID <- str_split_fixed(SelectionTablesShort,pattern = '.Table',n=2)[,1]

# Start with one file
SoundFilePathFull <- list.files(MaliauBoxDrive,full.names = T,recursive = T, pattern='.wav')

nslash <- str_count(SoundFilePathFull[1],'/')+1

SoundFilePathShort <- str_split_fixed(SoundFilePathFull,
                                      pattern = '/',n=nslash)[,nslash]

SoundFilePathShort <- str_split_fixed(SoundFilePathShort,
                                      pattern = '.wav',n=2)[,1]

AnnotationsPathFull <- SelectionTables


AnnotationsPathShort <- SelectionTablesID

for( b in 1: length(AnnotationsPathFull)){tryCatch({
  print(b)
  WavIndex <- which(str_detect(SoundFilePathShort,AnnotationsPathShort[b]))

  TempAnnotations <- read.delim2(AnnotationsPathFull[b])
  print(unique(TempAnnotations$Call.type))
  TempAnnotationsGibbon <- subset(TempAnnotations,Call.type=='female.gibbon')
  TempAnnotations <- subset(TempAnnotations,Call.type!='female.gibbon')

  if(nrow(TempAnnotationsGibbon) >0 ){
  TempWav <- readWave(SoundFilePathFull[WavIndex])
  WavDur <- duration(TempWav)

  Seq.start <- list()
  Seq.end <- list()

  i <- 1
  while (i + clip.duration < WavDur) {
    # print(i)
    Seq.start[[i]] = i
    Seq.end[[i]] = i+clip.duration
    i= i+hop.size
  }

  ClipStart <- unlist(Seq.start)
  ClipEnd <- unlist(Seq.end)

  ClipDataFrame <- cbind.data.frame(ClipStart,ClipEnd)

  TempClipsCombinedNoise <- data.frame()

  for(c in 1:nrow(TempAnnotationsGibbon)){

    TempRow <- TempAnnotationsGibbon[c,]

    StartTime <- as.numeric(TempRow$Begin.Time..s.)
    EndTime <- as.numeric(TempRow$End.Time..s.)

    # Duration of each clip
    ClipDataFrame$Duration <- ClipDataFrame$ClipEnd - ClipDataFrame$ClipStart

    # Duration of the overlap for each clip with the window
    ClipDataFrame$OverlapDuration <- pmin(ClipDataFrame$ClipEnd, EndTime) - pmax(ClipDataFrame$ClipStart, StartTime)

    # Only select clips where overlap duration is >= 2/3 of the clip's duration
    TempClips <- ClipDataFrame[which(ClipDataFrame$OverlapDuration >= overlap_threshold * ClipDataFrame$Duration &
                                       ClipDataFrame$OverlapDuration > 0),]



    TempClass <- 'Gibbons'

    if(nrow(TempClips) >0){

      subset.directory <- paste('/Volumes/DJC Files/Clink et al Zenodo Data/TestClipsMaliau/',TempClass,sep='')

      if (!dir.exists(subset.directory)){
        dir.create(subset.directory)
        print(paste('Created output dir',subset.directory))
      } else {
        print(paste(subset.directory,'already exists'))
      }
      short.sound.files <- lapply(1:nrow(TempClips),
                                  function(i)
                                    extractWave(
                                      TempWav,
                                      from = TempClips$ClipStart[i],
                                      to = TempClips$ClipEnd[i],
                                      xunit = c("time"),
                                      plot = F,
                                      output = "Wave"
                                    ))

      short.sound.files <- lapply(1:length(short.sound.files),
                                  function(i)
                                    downsample(
                                      short.sound.files[[i]],16000
                                    ))

      for(d in 1:length(short.sound.files)){

        writeWave(short.sound.files[[d]],paste(subset.directory,'/',
                                               TempClass,'_',AnnotationsPathShort[b],'_',TempClips$ClipEnd[d], '.wav', sep=''),
                  extensible = F)
      }

    }
  }

}

  if(nrow(TempAnnotations)>0) {

  TempWav <- readWave(SoundFilePathFull[WavIndex])
  WavDur <- duration(TempWav)

  for(c in 1:nrow(TempAnnotations)){

    TempRow <- TempAnnotations[c,]

    StartTime <- as.numeric(TempRow$Begin.Time..s.)
    EndTime <- as.numeric(TempRow$End.Time..s.)


    # Duration of each clip
    ClipDataFrame$Duration <- ClipDataFrame$ClipEnd - ClipDataFrame$ClipStart

    # Duration of the overlap for each clip with the window
    ClipDataFrame$OverlapDuration <- pmin(ClipDataFrame$ClipEnd, EndTime) - pmax(ClipDataFrame$ClipStart, StartTime)

    # Only select clips where overlap duration is >= 2/3 of the clip's duration
    TempClips <- ClipDataFrame[which(ClipDataFrame$OverlapDuration >= overlap_threshold * ClipDataFrame$Duration &
                                       ClipDataFrame$OverlapDuration > 0),]

    # Plus add 10 random clips
    set.seed(13)

    TempClass <- 'Noise'

    if(nrow(TempClips) >0){

      subset.directory <- paste('/Volumes/DJC Files/Clink et al Zenodo Data/TestClipsMaliau/',TempClass,sep='')

      if (!dir.exists(subset.directory)){
        dir.create(subset.directory)
        print(paste('Created output dir',subset.directory))
      } else {
        print(paste(subset.directory,'already exists'))
      }
      short.sound.files <- lapply(1:nrow(TempClips),
                                  function(i)
                                    extractWave(
                                      TempWav,
                                      from = TempClips$ClipStart[i],
                                      to = TempClips$ClipEnd[i],
                                      xunit = c("time"),
                                      plot = F,
                                      output = "Wave"
                                    ))

      short.sound.files <- lapply(1:length(short.sound.files),
                                  function(i)
                                    downsample(
                                      short.sound.files[[i]],16000
                                    ))

      for(d in 1:length(short.sound.files)){

        writeWave(short.sound.files[[d]],paste(subset.directory,'/',
                                               TempClass,'_',AnnotationsPathShort[b],'_',TempClips$ClipEnd[d], '.wav', sep=''),
                  extensible = F)
      }

    }

  }
}
}, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
}



