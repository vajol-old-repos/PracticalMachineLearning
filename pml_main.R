## Set working directory
setwd("c:\\")
if (!file.exists("tmp")) {
  dir.create("tmp")
}
setwd("c:\\tmp")

## Set URL for download
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

## Download the archive file
download.file(fileUrl, "pml-training.csv")

## Document when the file has been downloaded
dateDownloaded <- date()
dateDownloaded

## Load the data into table and label all NA values correctly
origData <- read.table("pml-training.csv", header = TRUE, sep = ",", fill = TRUE, na.strings=c("NA","","#DIV/0!"))

## Show number of rows and columns
dim(origData)

## Show the structure of the data set
str(origData)

# Remove non-relevant data from the data set
nonRelevant <- grep("X|timestamp|user_name|new_window|num_window", names(origData))
origData <- origData[,-nonRelevant]

# Remove all columns containing NA values
cleanData <- origData[,colSums(is.na(origData))==0]

## Show the structure of the new cleaned up data set
str(cleanData)

## Load "Caret" package
library(caret)

## Split data into training, test and validation sets
inTrain <- createDataPartition(y = cleanData$classe, p = 0.6, list=FALSE)
training <- cleanData[inTrain,]

tmp <- cleanData[-inTrain,]
inTrain <- createDataPartition(y = tmp$classe,  p =  0.5, list=FALSE)

test <- tmp[inTrain,]
validation <- tmp[-inTrain,]


## Preprocessing the data with PCA
ppData <- preProcess(training[,-53], method = 'pca', thresh = 0.99)

## Predicting from preprocessed data
trainingPredict <- predict(ppData, training[,-53])     
testPredict <- predict(ppData, test[,-53]) 
validationPredict <- predict(ppData, validation[,-53])    

# Apply SVM algorithm
modelSvm <- train(training$classe ~., data=trainingPredict, method='svmRadial')

# Apply RPART algorithm
modelRpart <- train(training$classe ~., data=trainingPredict, method="rpart")

# Apply GBM algorithm
modelGbm <- train(training$classe ~., data=trainingPredict, method="gbm", verbose=FALSE)

# Apply Random forest algorithm
trCont <- trainControl(method = "cv", number = 10)
modelRf <- train(training$classe ~., data=trainingPredict, method='rf', trControl = trCont)

# Apply LDA algorithm
modelLda <- train(training$classe ~., data=trainingPredict, method="lda")

predSvm <- predict(modelSvm,  validationPredict)
predRpart <- predict(modelRpart,  validationPredict)
predGbm <- predict(modelGbm,  validationPredict)
predRf <- predict(modelRf,  validationPredict)
predLda <- predict(modelLda,  validationPredict)

confusionMatrix(predSvm, validation$classe)$overall["Accuracy"]
Accuracy 
confusionMatrix(predRpart, validation$classe)$overall["Accuracy"]
Accuracy 
confusionMatrix(predGbm, validation$classe)$overall["Accuracy"]
Accuracy 
confusionMatrix(predRf, validation$classe)$overall["Accuracy"]
Accuracy 
confusionMatrix(predLda, validation$classe)$overall["Accuracy"] 
Accuracy 

## Predicting using test data set
taData <- read.table("pml-testing.csv", header = TRUE, sep = ",", fill = TRUE, na.strings=c("NA","","#DIV/0!"))
nrData <- grep("X|timestamp|user_name|new_window|num_window", names(taData))
taData <- taData[,-nrData]
ctaData <- taData[,colSums(is.na(taData))==0]

tp <- predict(ppData, ctaData[,-53])  

predictions <- predict(modelRf,  tp)
