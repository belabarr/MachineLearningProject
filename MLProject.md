# Machine Learning Project
Bel Abarrientos  

## Get Training and Test Data
Some codes were commented to reduce knitr details


```r
setwd("D:/$Study/DataScience/ML")
#if(!file.exists("./data")){dir.create("./data")}
#f1url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
setwd("data")
#download.file(f1url,destfile="pml-training.csv")

#f2url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#download.file(f2url,destfile="pml-testing.csv")

trainAll <- read.csv("pml-training.csv")
testAll <- read.csv("pml-testing.csv")
```

## Load required libraries


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(e1071)
```

## Explore Data using R Studio
Review of features and what can be considered as predictors, classes
Code commented to reduce details


```r
#head(testAll)
#head(trainAll)
#View(testAll)
#View(trainAll)
```

## Split Training data for model fitting
Derived model will be tested against the remaining training data.  
Further data exploration is done to assess predictors and decide which should be removed.  One review of test data, some columns are completely of NA values, so i decided to removed these columns from the training set.



```r
set.seed(1)
trains1 <- createDataPartition(y=trainAll$classe, p=0.1,list=FALSE)
train1d <- trainAll[trains1,]
train2d <- trainAll[-trains1,]

nzv <- nearZeroVar(train1d, saveMetrics=TRUE)
nzv
```

```
##                            freqRatio percentUnique zeroVar   nzv
## X                           1.000000   100.0000000   FALSE FALSE
## user_name                   1.054441     0.3054990   FALSE FALSE
## raw_timestamp_part_1        1.125000    38.6456212   FALSE FALSE
## raw_timestamp_part_2        1.500000    98.3197556   FALSE FALSE
## cvtd_timestamp              1.089041     1.0183299   FALSE FALSE
## new_window                 44.674419     0.1018330   FALSE  TRUE
## num_window                  1.125000    38.9511202   FALSE FALSE
## roll_belt                   1.067961    20.8248473   FALSE FALSE
## pitch_belt                  1.136364    42.7189409   FALSE FALSE
## yaw_belt                    1.226415    33.7576375   FALSE FALSE
## total_accel_belt            1.134367     1.2219959   FALSE FALSE
## kurtosis_roll_belt       1921.000000     2.2403259   FALSE  TRUE
## kurtosis_picth_belt       320.166667     1.9348269   FALSE  TRUE
## kurtosis_yaw_belt          44.674419     0.1018330   FALSE  TRUE
## skewness_roll_belt       1921.000000     2.2403259   FALSE  TRUE
## skewness_roll_belt.1      320.166667     1.9857434   FALSE  TRUE
## skewness_yaw_belt          44.674419     0.1018330   FALSE  TRUE
## max_roll_belt               1.500000     1.8839104   FALSE FALSE
## max_picth_belt              1.000000     0.6109980   FALSE FALSE
## max_yaw_belt              384.200000     1.4256619   FALSE  TRUE
## min_roll_belt               1.000000     1.8839104   FALSE FALSE
## min_pitch_belt              1.125000     0.5091650   FALSE FALSE
## min_yaw_belt              384.200000     1.4256619   FALSE  TRUE
## amplitude_roll_belt         1.400000     1.3747454   FALSE FALSE
## amplitude_pitch_belt        1.333333     0.4073320   FALSE FALSE
## amplitude_yaw_belt         46.853659     0.2036660   FALSE  TRUE
## var_total_accel_belt        1.111111     0.6109980   FALSE FALSE
## avg_roll_belt               2.500000     1.7311609   FALSE FALSE
## stddev_roll_belt            1.166667     0.9164969   FALSE FALSE
## var_roll_belt               1.875000     0.8146640   FALSE FALSE
## avg_pitch_belt              1.500000     1.8329939   FALSE FALSE
## stddev_pitch_belt           1.428571     0.8655804   FALSE FALSE
## var_pitch_belt              1.181818     0.8146640   FALSE FALSE
## avg_yaw_belt                1.000000     1.9857434   FALSE FALSE
## stddev_yaw_belt             2.600000     0.9164969   FALSE FALSE
## var_yaw_belt                1.333333     1.3238289   FALSE FALSE
## gyros_belt_x                1.071429     4.2769857   FALSE FALSE
## gyros_belt_y                1.275862     2.4439919   FALSE FALSE
## gyros_belt_z                1.061798     6.0081466   FALSE FALSE
## accel_belt_x                1.023810     6.3645621   FALSE FALSE
## accel_belt_y                1.031056     5.4989817   FALSE FALSE
## accel_belt_z                1.177778     9.6232179   FALSE FALSE
## magnet_belt_x               1.205128     9.5723014   FALSE FALSE
## magnet_belt_y               1.301587     8.9613035   FALSE FALSE
## magnet_belt_z               1.140000    13.1364562   FALSE FALSE
## roll_arm                   38.777778    51.2729124   FALSE FALSE
## pitch_arm                  49.857143    53.0549898   FALSE FALSE
## yaw_arm                    31.727273    53.6659878   FALSE FALSE
## total_accel_arm             1.086957     2.9531568   FALSE FALSE
## var_accel_arm               1.000000     2.1894094   FALSE FALSE
## avg_roll_arm                8.000000     1.8329939   FALSE FALSE
## stddev_roll_arm             8.000000     1.8329939   FALSE FALSE
## var_roll_arm                8.000000     1.8329939   FALSE FALSE
## avg_pitch_arm               8.000000     1.8329939   FALSE FALSE
## stddev_pitch_arm            8.000000     1.8329939   FALSE FALSE
## var_pitch_arm               8.000000     1.8329939   FALSE FALSE
## avg_yaw_arm                 8.000000     1.8329939   FALSE FALSE
## stddev_yaw_arm              8.000000     1.8329939   FALSE FALSE
## var_yaw_arm                 8.000000     1.8329939   FALSE FALSE
## gyros_arm_x                 1.169811    25.6619145   FALSE FALSE
## gyros_arm_y                 1.687500    14.5621181   FALSE FALSE
## gyros_arm_z                 1.145455     9.2668024   FALSE FALSE
## accel_arm_x                 1.047619    28.8187373   FALSE FALSE
## accel_arm_y                 1.476190    21.4867617   FALSE FALSE
## accel_arm_z                 1.000000    27.6476578   FALSE FALSE
## magnet_arm_x                1.090909    45.0101833   FALSE FALSE
## magnet_arm_y                1.000000    34.9796334   FALSE FALSE
## magnet_arm_z                1.769231    40.3258656   FALSE FALSE
## kurtosis_roll_arm         240.125000     1.8839104   FALSE  TRUE
## kurtosis_picth_arm        240.125000     1.8839104   FALSE  TRUE
## kurtosis_yaw_arm         1921.000000     2.2403259   FALSE  TRUE
## skewness_roll_arm         240.125000     1.8839104   FALSE  TRUE
## skewness_pitch_arm        240.125000     1.8839104   FALSE  TRUE
## skewness_yaw_arm         1921.000000     2.2403259   FALSE  TRUE
## max_roll_arm                4.000000     1.7820774   FALSE FALSE
## max_picth_arm               8.000000     1.8329939   FALSE FALSE
## max_yaw_arm                 1.333333     1.3747454   FALSE FALSE
## min_roll_arm                4.000000     1.7820774   FALSE FALSE
## min_pitch_arm               8.000000     1.8329939   FALSE FALSE
## min_yaw_arm                 1.000000     1.2219959   FALSE FALSE
## amplitude_roll_arm          4.000000     1.7311609   FALSE FALSE
## amplitude_pitch_arm         8.000000     1.8329939   FALSE FALSE
## amplitude_yaw_arm           1.666667     1.4765784   FALSE FALSE
## roll_dumbbell               1.500000    94.3991853   FALSE FALSE
## pitch_dumbbell              1.285714    92.5152749   FALSE FALSE
## yaw_dumbbell                1.500000    93.9409369   FALSE FALSE
## kurtosis_roll_dumbbell   1921.000000     2.2403259   FALSE  TRUE
## kurtosis_picth_dumbbell  1921.000000     2.2403259   FALSE  TRUE
## kurtosis_yaw_dumbbell      44.674419     0.1018330   FALSE  TRUE
## skewness_roll_dumbbell   1921.000000     2.2403259   FALSE  TRUE
## skewness_pitch_dumbbell  1921.000000     2.2403259   FALSE  TRUE
## skewness_yaw_dumbbell      44.674419     0.1018330   FALSE  TRUE
## max_roll_dumbbell           1.000000     2.1894094   FALSE FALSE
## max_picth_dumbbell          1.000000     2.1894094   FALSE FALSE
## max_yaw_dumbbell          640.333333     1.6293279   FALSE  TRUE
## min_roll_dumbbell           2.000000     2.1384929   FALSE FALSE
## min_pitch_dumbbell          2.000000     2.1384929   FALSE FALSE
## min_yaw_dumbbell          640.333333     1.6293279   FALSE  TRUE
## amplitude_roll_dumbbell     2.000000     2.1384929   FALSE FALSE
## amplitude_pitch_dumbbell    1.000000     2.1894094   FALSE FALSE
## amplitude_yaw_dumbbell     44.674419     0.1018330   FALSE  TRUE
## total_accel_dumbbell        1.021277     2.0366599   FALSE FALSE
## var_accel_dumbbell          2.000000     2.1384929   FALSE FALSE
## avg_roll_dumbbell           1.000000     2.1894094   FALSE FALSE
## stddev_roll_dumbbell        1.000000     2.1894094   FALSE FALSE
## var_roll_dumbbell           1.000000     2.1894094   FALSE FALSE
## avg_pitch_dumbbell          1.000000     2.1894094   FALSE FALSE
## stddev_pitch_dumbbell       1.000000     2.1894094   FALSE FALSE
## var_pitch_dumbbell          1.000000     2.1894094   FALSE FALSE
## avg_yaw_dumbbell            1.000000     2.1894094   FALSE FALSE
## stddev_yaw_dumbbell         1.000000     2.1894094   FALSE FALSE
## var_yaw_dumbbell            1.000000     2.1894094   FALSE FALSE
## gyros_dumbbell_x            1.140351     8.4012220   FALSE FALSE
## gyros_dumbbell_y            1.343750     9.7759674   FALSE FALSE
## gyros_dumbbell_z            1.029412     6.7209776   FALSE FALSE
## accel_dumbbell_x            1.000000    13.4928717   FALSE FALSE
## accel_dumbbell_y            1.416667    18.3299389   FALSE FALSE
## accel_dumbbell_z            1.064516    16.1405295   FALSE FALSE
## magnet_dumbbell_x           1.136364    28.8187373   FALSE FALSE
## magnet_dumbbell_y           1.611111    29.1751527   FALSE FALSE
## magnet_dumbbell_z           1.166667    24.5417515   FALSE FALSE
## roll_forearm                9.200000    30.6008147   FALSE FALSE
## pitch_forearm              36.800000    48.9307536   FALSE FALSE
## yaw_forearm                13.629630    32.9429735   FALSE FALSE
## kurtosis_roll_forearm     240.125000     1.8329939   FALSE  TRUE
## kurtosis_picth_forearm    213.444444     1.8329939   FALSE  TRUE
## kurtosis_yaw_forearm       44.674419     0.1018330   FALSE  TRUE
## skewness_roll_forearm     240.125000     1.8839104   FALSE  TRUE
## skewness_pitch_forearm    213.444444     1.8329939   FALSE  TRUE
## skewness_yaw_forearm       44.674419     0.1018330   FALSE  TRUE
## max_roll_forearm            8.000000     1.8329939   FALSE FALSE
## max_picth_forearm           2.000000     1.4765784   FALSE FALSE
## max_yaw_forearm           240.125000     1.0183299   FALSE  TRUE
## min_roll_forearm            8.000000     1.8329939   FALSE FALSE
## min_pitch_forearm           2.000000     1.5274949   FALSE FALSE
## min_yaw_forearm           240.125000     1.0183299   FALSE  TRUE
## amplitude_roll_forearm      8.000000     1.8329939   FALSE FALSE
## amplitude_pitch_forearm     2.250000     1.5274949   FALSE FALSE
## amplitude_yaw_forearm      54.885714     0.1527495   FALSE  TRUE
## total_accel_forearm         1.133333     3.0040733   FALSE FALSE
## var_accel_forearm           1.000000     2.1894094   FALSE FALSE
## avg_roll_forearm            8.000000     1.8329939   FALSE FALSE
## stddev_roll_forearm         8.000000     1.8329939   FALSE FALSE
## var_roll_forearm            8.000000     1.8329939   FALSE FALSE
## avg_pitch_forearm           8.000000     1.8329939   FALSE FALSE
## stddev_pitch_forearm        8.000000     1.8329939   FALSE FALSE
## var_pitch_forearm           8.000000     1.8329939   FALSE FALSE
## avg_yaw_forearm             8.000000     1.8329939   FALSE FALSE
## stddev_yaw_forearm          9.000000     1.7820774   FALSE FALSE
## var_yaw_forearm             9.000000     1.7820774   FALSE FALSE
## gyros_forearm_x             1.037736    10.9979633   FALSE FALSE
## gyros_forearm_y             1.441176    28.2586558   FALSE FALSE
## gyros_forearm_z             1.038462    10.7433809   FALSE FALSE
## accel_forearm_x             1.076923    32.5356415   FALSE FALSE
## accel_forearm_y             1.416667    35.0814664   FALSE FALSE
## accel_forearm_z             1.157895    21.1303462   FALSE FALSE
## magnet_forearm_x            1.181818    43.6863544   FALSE FALSE
## magnet_forearm_y            1.166667    48.8289206   FALSE FALSE
## magnet_forearm_z            1.222222    46.1812627   FALSE FALSE
## classe                      1.468421     0.2545825   FALSE FALSE
```

## Set-up data using Predictor Variables only
NA columns are removed or those with UniquePercent almost zero


```r
testAllNA <- testAll[,c(12:36, 50:59, 69:83,87:101, 103:112, 125:139, 141:150)]
testAllNoNA <- testAll[,-c(12:36, 50:59, 69:83,87:101, 103:112, 125:139, 141:150)]
colNa <- colnames(testAllNA)

train1dNoNA <- train1d[,!(names(train1d) %in% colNa)]
train2dNoNA <- train2d[,!(names(train2d) %in% colNa)]
testNoNA <- testAll[,!(names(testAll) %in% colNa)]

train1fl <- train1dNoNA[8:60]
train2fl <- train2dNoNA[8:60]
testfl   <-  testNoNA[8:59]
```

## Model Fit using subset of training data and randomForest
various size of predictor variables were tested from 5, 8, 10, 12, 14, 15, 16, 20, 40.  mtry=12 provide the best accuracy result 

At this number of predictors, the result is 95.76% accurate, out of sample error is very minimal 0.11% the most (class B).  Overall dataset out-of-bag error is 5.04%. 




```r
set.seed(2)
trainmod <- randomForest(formula = classe~.,data=train1fl, mtry=12, importance=TRUE)
print(trainmod)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train1fl, mtry = 12,      importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 12
## 
##         OOB estimate of  error rate: 5.04%
## Confusion matrix:
##     A   B   C   D   E class.error
## A 547   2   2   7   0  0.01971326
## B  13 337  23   3   4  0.11315789
## C   0  17 321   5   0  0.06413994
## D   2   2  10 308   0  0.04347826
## E   1   1   4   3 352  0.02493075
```

```r
testmod <- predict(trainmod, newdata=train2fl)
confusionMatrix(testmod,train2fl$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4880  110    6   14    2
##          B   31 3177   81    0   22
##          C   25  109 2927   94   31
##          D   63   19   62 2771   36
##          E   23    2    3   15 3155
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9576          
##                  95% CI : (0.9546, 0.9606)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9464          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9717   0.9298   0.9506   0.9575   0.9720
## Specificity            0.9896   0.9906   0.9822   0.9878   0.9970
## Pos Pred Value         0.9737   0.9595   0.9187   0.9390   0.9866
## Neg Pred Value         0.9888   0.9833   0.9895   0.9916   0.9937
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2764   0.1799   0.1658   0.1569   0.1787
## Detection Prevalence   0.2838   0.1875   0.1804   0.1671   0.1811
## Balanced Accuracy      0.9806   0.9602   0.9664   0.9727   0.9845
```

```r
testClasse <- predict(trainmod, newdata=testfl)
testClasse
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  D  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

## Compare random forests with 10-fold cross validation
Support Vendor Machines (SVM), ten repititions of 10-fold cross-validation using errorest functions in ipred package is compared with randomForest.
Code commented due to lengthy processing.

RandomForest error rate is consistent with SVM error rate using small training model, rf mean 0.5229 , SVM  mean 0.5662.  


```r
library(ipred)
set.seed(12)
ptm <- proc.time()
error.RF <- numeric(10) 
#for (i in 1:10) {
#    error.RF[i] <- errorest(classe ~., data=train1fl, model=randomForest, mtry=12)$error
#    }
proc.time() - ptm
```

```
##    user  system elapsed 
##       0       0       0
```

```r
#summary(error.RF)

set.seed(112)
error.SVM <- numeric(10) 
#for (i in 1:10) {
#    error.SVM[i] <- errorest(classe ~., data=train1fl, model=svm, cost=10, gamma=1.5)$error
#    }
#summary(error.SVM)
```

## Generate the submission files


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(testClasse)
```
Citation:
Data used in this assignment is from Human Activity Recognition by:
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
http://groupware.les.inf.puc-rio.br/har
