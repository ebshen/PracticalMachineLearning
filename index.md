# Practical Machine Learning - Course Project
Eric Shen  
October 29, 2017  



## 1. Overview

In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to predict the manner in which they did the exercise (the "classe" variable in the training set). The participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Finally, we will test the developed prediction model to predict 20 different test cases.

Acknowledgement:
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. http://groupware.les.inf.puc-rio.br/har

## 2. Data Preprocessing

The following code loads the data and required libraries. The original dataset contains 160 variables (including classe). We reduce the number of variables through three filtering steps: 1) removing variables that are mostly (> 90%) NA, 2) removing variables that have near zero variance, and 3) removing identifier variables not relevant for prediction.

The "training" set is split 70/30 into "TrainSet"" and "TestSet"" for model training and testing. The "testing" set is reserved for the final model test on 20 cases.


```r
# read the data
library(caret, quietly = TRUE)
library(parallel, quietly = TRUE)
library(doParallel, quietly = TRUE)
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv") # for final testing

# partition the training set to develop models
set.seed(111)
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain,]
TestSet <- training[-inTrain,]

# filter
# remove variables with more than 90% NAs
NAfilter <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.90
TrainSet <- TrainSet[, NAfilter==FALSE]
TestSet <- TestSet[, NAfilter==FALSE]

# remove variables with near zero variance
NZVfilter <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZVfilter]
TestSet <- TestSet[, -NZVfilter]

# remove first 6 variables
TrainSet <- TrainSet[, -(1:6)]
TestSet <- TestSet[, -(1:6)]

# final variables
names(TrainSet)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

```r
# split into x and y
x <- TrainSet[,-53]
y <- TrainSet[,53]
```

The final 52 variables are listed above. We also split "TrainSet" into x (without classe variable) and y (only the classe variable).

## 3. Prediction Models, Cross-Validation, Out of sample Accuracy

Three models are trained and tested: 1) Decision tree (rpart), 2) Random forests (rf), and 3) Gradient boosting machine (gbm).

The following instructions from
https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md were used to enable parallel processing and 3-fold cross validation. The following are the out of sample results for each model (from evaluation on TestSet).

### Decision Tree (rpart)


```r
cluster <- makeCluster(detectCores() - 1)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
```


```r
fitControl <- trainControl(method = "cv", number = 3, allowParallel = TRUE)
set.seed(111)
modelFitRpart <- train(x,y, method = "rpart", trControl = fitControl)
Rpartresult <- predict(modelFitRpart, TestSet)
confusionMatrix(TestSet$classe, Rpartresult)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.4966865
```

### Random Forest


```r
modelFitRf <- train(x , y, method = "rf", trControl = fitControl)
rfresult <- predict(modelFitRf, TestSet)
confusionMatrix(TestSet$classe, rfresult)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9915038
```

### Gradient Boosting Machine


```r
modelFitGbm <- train(x,y, method = "gbm", trControl = fitControl, verbose = FALSE)
Gbmresult <- predict(modelFitGbm, TestSet)
confusionMatrix(TestSet$classe, Gbmresult)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9607477
```


```r
stopCluster(cluster)
registerDoSEQ()
```

From the out of sample accuracy results, we choose the Random Forest model based on the 99.15% accuracy.

## 4. Final model testing

The final model (Random Forest) is applied to the "testing" set of 20 cases for final evaluation.


```r
predictTesting <- predict(modelFitRf, newdata = testing)
predictTesting
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
