require(devtools)

buildBD <- function(fortranChange=F,cartridgeChange=F,cartridges,build=T,pdf=F,check=F){
  require(rjson)
  currwd <- getwd()
  OS <- getOS()
  if(fortranChange){
    #only do this part on Mac
    if(OS==1){
      setwd("~/LAPK/BestDoseSource/Source")
      #combine all the fortran source files
      system("cat *.FOR > bestdose.for")
      # #remove comment lines and clean up
      # rmComm <- function(files){
      #   for (i in files){
      #     system(paste("~/LAPK/PmetricsSource/Source/win2mac.sa",i))
      #     file.remove(i)
      #     file.rename("newfile.txt",i)
      #     code <- readLines(i)
      #     commLines <- grep("^C",code,ignore.case=T)
      #     commLines2 <- grep("^\\*",code)
      #     if(length(commLines2)>0) {code <- code[-c(commLines,commLines2)]} else {code <- code[-commLines]}
      #     code <- code[code!=""]
      #     writeLines(code,i,sep="\r\n")
      #   }
      # }
      # 
      # rmComm(files="bestdose.for")
    }
    
  }#continue on both Mac/Windows
  
  setwd(c("~/LAPK/BestDoseSource",
          "C:/LAPK/BestDoseSource")[OS])  
  document("BestDose")
  if(pdf){
    file.remove("./BestDose/inst/doc/BestDose-manual.pdf")
    if(OS==1){
      system("R CMD Rd2pdf --output=./BestDose/inst/doc/BestDose-manual.pdf --no-preview ./BestDose/man")
    }
    if(OS==2){
      shell("R CMD Rd2pdf --output=.\\BestDose\\inst\\doc\\BestDose-manual.pdf --no-preview .\\BestDose\\man")
    }
  }
  
  if(check) check("BestDose")
  if(build){
    build("BestDose",binary=T)
    build("BestDose",binary=F)
  } 
  install("BestDose")
  
  
  
  #nextline function
  getNext <- function(build){
    return(length(build)+1)
  }
  
  if(cartridgeChange){
    #loop through the drugs and make the doser for each
    setwd(c("~/LAPK/BestDoseSource",
            "C:/LAPK/BestDoseSource")[OS])    
    drugFolders <- list.files("Drugs",pattern=cartridges)
    for(i in drugFolders){
      setwd(c(paste("~/LAPK/BestDoseSource/Drugs",i,sep="/"),
              paste("C:/LAPK/BestDoseSource/Drugs",i,sep="/"))[OS])
      meta <- fromJSON(file="meta.json")
      
      #write the bd.tmpl template
      bdTemp <- vector("character")
      bdTemp[getNext(bdTemp)] <- "BESTDOS OCT_13"
      bdTemp[getNext(bdTemp)] <- " MODEL FILENAME"
      bdTemp[getNext(bdTemp)] <- "model.for"
      bdTemp[getNext(bdTemp)] <- " IRAN INDICES"
      bdTemp[getNext(bdTemp)] <- meta$model$iran
      bdTemp[getNext(bdTemp)] <- " NPAG DENSITY FILE"
      bdTemp[getNext(bdTemp)] <- "{{.DenFile}}"
      bdTemp[getNext(bdTemp)] <- " MAXCYC"
      bdTemp[getNext(bdTemp)] <- "{{.MaxCy}}"
      bdTemp[getNext(bdTemp)] <- " INCLUDPAST"
      bdTemp[getNext(bdTemp)] <- "{{.IncludePast}}"
      bdTemp[getNext(bdTemp)] <- " IPASTFILE"
      bdTemp[getNext(bdTemp)] <- "{{.IPastFile}}"
      bdTemp[getNext(bdTemp)] <- " PASTFILEIN"
      bdTemp[getNext(bdTemp)] <- "{{.PastCsv}}"
      bdTemp[getNext(bdTemp)] <- " ICSVFILE"
      bdTemp[getNext(bdTemp)] <- "1"
      bdTemp[getNext(bdTemp)] <- " FUTUREFILEIN"
      bdTemp[getNext(bdTemp)] <- "{{.FutCsv}}"
      bdTemp[getNext(bdTemp)] <- " TNEXT"
      bdTemp[getNext(bdTemp)] <- "{{.TNext}}"
      bdTemp[getNext(bdTemp)] <- " IDELTA"
      bdTemp[getNext(bdTemp)] <- "{{.IDelta}}"
      bdTemp[getNext(bdTemp)] <- " NOFIX"
      bdTemp[getNext(bdTemp)] <- meta$model$nofix
      bdTemp[getNext(bdTemp)] <- " VALFIX ARRAY IF NOFIX > 0"
      bdTemp[getNext(bdTemp)] <- ifelse(length(meta$model$valfix)==0,NA,paste(unlist(strsplit(meta$model$valfix,",")),collapse="\n"))
      bdTemp[getNext(bdTemp)] <- " TOLER"
      bdTemp[getNext(bdTemp)] <- "  1.00000000000000005E-004"
      bdTemp[getNext(bdTemp)] <- " NUMEQT"
      bdTemp[getNext(bdTemp)] <- meta$model$numeqt
      bdTemp[getNext(bdTemp)] <- "  NUMEQT LINES OF ASSAY COEFFICIENTS"
      bdTemp[getNext(bdTemp)] <- paste(unlist(meta$outputs),collapse="\n")
      bdTemp[getNext(bdTemp)] <- " IERRMOD"
      bdTemp[getNext(bdTemp)] <- meta$model$errormod$ierrmod
      bdTemp[getNext(bdTemp)] <- " GAMLAM"
      bdTemp[getNext(bdTemp)] <- "{{.GamLam}}"
      bdTemp[getNext(bdTemp)] <- "  IASS(I),I=1,NUMEQT"
      bdTemp[getNext(bdTemp)] <- rep(1,meta$model$numeqt,collapse="   ")
      bdTemp[getNext(bdTemp)] <- " NDRUG"
      bdTemp[getNext(bdTemp)] <- meta$model$inputs
      bdTemp[getNext(bdTemp)] <- "  AF(I),I=1,NDRUG"
      bdTemp[getNext(bdTemp)] <- paste(unlist(sapply(meta$drugs,function(x) x$salt)),collapse="   ")
      bdTemp[getNext(bdTemp)] <- " IOPTIMIZE"
      bdTemp[getNext(bdTemp)] <- "{{.IOpt}}"
      bdTemp[getNext(bdTemp)] <- " BIASWEIGHT"
      bdTemp[getNext(bdTemp)] <- "{{.BiasWeight}}"
      bdTemp[getNext(bdTemp)] <- " ITARGET"
      bdTemp[getNext(bdTemp)] <- "{{.ITarget}}"
      
      bdTemp <- bdTemp[!is.na(bdTemp)]
      writeLines(bdTemp,"bd.tmpl")
      if(file.exists("bd.inx")) file.remove("bd.inx") #update to new format
      
      
      
      #write EXTNUM
      writeLines("    1","EXTNUM")
      
      #compile cartridge engine
      cat(paste("Compiling ",meta$model$name," cartridge...\n",sep=""))
      flush.console()
      if(OS==1){
        system("gfortran -O3 -m64 -o bestdose.exe ../../Source/bestdose.for model.for",ignore.stdout=T,ignore.stderr=T)
      }
      if(OS==2){
        shell("gfortran -O3 -m64 -o bestdose.exe ..\\..\\Source\\bestdose.for model.for",ignore.stdout=T,ignore.stderr=T) 
      }
      
      #make zipped archive
      setwd(c("~/LAPK/BestDoseSource/Drugs",
              "C:/LAPK/BestDoseSource/Drugs")[OS])
      zip(c(paste("~/LAPK/BestDoseSource/ZippedDrugs/",i,sep=""),
            paste("C:/LAPK/BestDoseSource/WinZippedDrugs/",i,sep=""))[OS],i)
    }
    
    
    
  } #end if cartridgeChange
  
  #install the drug models to my copy of the package
  
  drugFolders <- list.files(c("~/LAPK/BestDoseSource/ZippedDrugs",
                              "C:/LAPK/BestDoseSource/WinZippedDrugs")[OS],full.names=T)
  for(thisCart in drugFolders){
    install.BDcartridge(thisCart,overwrite=T)
  }
  
  #restore working directory
  setwd(currwd)
} #end function


#use regular expression for cartridges argument
buildBD(fortranChange=F,cartridgeChange=T,cartridges=c("Cefepime"),build=T,pdf=F,check=F)
buildBD(fortranChange=T,cartridgeChange=F,cartridges=c("Cefepime"),build=F,pdf=F,check=F)




