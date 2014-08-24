# Practical Machine Learning course project assignment
Vajo Lukic  
Sunday, August 24, 2014  


Human Activity Recognition
==============================================================================================


Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The goal of the project is to predict the manner in which they did the exercise.

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har


Loading the data
--------------------------------------------------------------
Set working directory:

```r
setwd("c:\\")
if (!file.exists("tmp")) {
  dir.create("tmp")
}
setwd("c:\\tmp")
```

Set URL for download:

```r
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
```

Download the archive file:

```r
download.file(fileUrl, "pml-training.csv")
```

Document when the file has been downloaded:

```r
dateDownloaded <- date()
dateDownloaded
```

Load the data into table and label all NA values correctly:

```r
origData <- read.table("pml-training.csv", header = TRUE, sep = ",", fill = TRUE, na.strings=c("NA","","#DIV/0!"))
```


Cleaning up the data
--------------------------------------------------------------
Remove non-relevant columns from the data set:

```r
nonRelevant <- grep("X|timestamp|user_name|new_window|num_window", names(origData))
origData <- origData[,-nonRelevant]
```

Remove all columns containing NA values:

```r
cleanData <- origData[,colSums(is.na(origData))==0]
```


Preprocessing the data
--------------------------------------------------------------
Split data into training, test and validation sets (training 60%, test 20%, validation 20%):

```r
inTrain <- createDataPartition(y = cleanData$classe, p = 0.6, list=FALSE)
training <- cleanData[inTrain,]

tmp <- cleanData[-inTrain,]
inTrain <- createDataPartition(y = tmp$classe,  p =  0.5, list=FALSE)

test <- tmp[inTrain,]
validation <- tmp[-inTrain,]
```

Preprocessing the data with Principal Components Analysis (PCA):

```r
ppData <- preProcess(training[,-53], method = 'pca', thresh = 0.99)
```

Reducing number of predictors:

```r
trainingPredict <- predict(ppData, training[,-53])     
testPredict <- predict(ppData, test[,-53]) 
validationPredict <- predict(ppData, validation[,-53])    
```


Training the models
--------------------------------------------------------------
Apply Support vector machines (SVM) analysis:

```r
modelSvm <- train(training$classe ~., data=trainingPredict, method='svmRadial')
```

Apply Recursive Partitioning and Regression Trees (RPART) analysis:

```r
modelRpart <- train(training$classe ~., data=trainingPredict, method="rpart")
```

Apply Naive Bayes (GBM) analysis:

```r
modelGbm <- train(training$classe ~., data=trainingPredict, method="gbm", verbose=FALSE)
```

Apply Random forests (RF) analysis:

```r
trCont <- trainControl(method = "cv", number = 10)
modelRf <- train(training$classe ~., data=trainingPredict, method='rf', trControl = trCont)
```

Apply Linear discriminant analysis (LDA) analysis:

```r
modelLda <- train(training$classe ~., data=trainingPredict, method="lda")
```


Cross validate analysis results by using validation data set: 

```r
predSvm <- predict(modelSvm,  validationPredict)
predRpart <- predict(modelRpart,  validationPredict)
predGbm <- predict(modelGbm,  validationPredict)
predRf <- predict(modelRf,  validationPredict)
predLda <- predict(modelLda,  validationPredict)
```


Get "accuracy" from each model:

```r
confusionMatrix(predSvm, validation$classe)$overall["Accuracy"]
Accuracy 
0.9395871
```


```r
confusionMatrix(predRpart, validation$classe)$overall["Accuracy"]
Accuracy 
0.3782819
```


```r
confusionMatrix(predGbm, validation$classe)$overall["Accuracy"]
Accuracy 
0.8455264 
```


```r
confusionMatrix(predRf, validation$classe)$overall["Accuracy"]
Accuracy 
0.972725 
```


```r
confusionMatrix(predLda, validation$classe)$overall["Accuracy"] 
Accuracy 
0.591894
```



Final conclusion
--------------------------------------------------------------
Clearly, "Random Forest" model has produced the best reusults, with accuracy of 97,27% on validation data set. We'll choose this model as our best, final model. In that case, expected "Out of sample" error is 2,7275 %.



Final prediction
--------------------------------------------------------------

```r
taData <- read.table("pml-testing.csv", header = TRUE, sep = ",", fill = TRUE, na.strings=c("NA","","#DIV/0!"))
nrData <- grep("X|timestamp|user_name|new_window|num_window", names(taData))
taData <- taData[,-nrData]
ctaData <- taData[,colSums(is.na(taData))==0]

tp <- predict(ppData, ctaData[,-53])  

predictions <- predict(modelRf,  tp)
predictions
## B A B A A C D B A A B C B A E E A B B B
## Levels: A B C D E
```


