---
title: "Squareroot Prediction"
author: "Riya Patel"
date: "22/07/2020"
output: pdf_document
---



## Introduction
We are going to use the data set of squareroots generated in excel to make predictions using the neural network model.The goal of this analysis is to train a neural network to predict the square root of number 810 and compare it with the actual square root value.

## Data Analysis

The data set must be imported and a data set must be created from the CSV. The columns used are
* number
* sqrt
 for the square roots as shown below:


```r
setwd("C:/Users/User/Desktop/SUMMER 2020/APT3010")
myData <- read.csv("squareroots.csv")
names(myData)<- c("number","sqrt")

head(myData)
```

```
##   number     sqrt
## 1      1 1.000000
## 2      2 1.414214
## 3      3 1.732051
## 4      4 2.000000
## 5      5 2.236068
## 6      6 2.449490
```

```r
tail(myData)
```

```
##     number     sqrt
## 800    800 28.28427
## 801    801 28.30194
## 802    802 28.31960
## 803    803 28.33725
## 804    804 28.35489
## 805    805 28.37252
```

```r
summary(myData)
```

```
##      number         sqrt      
##  Min.   :  1   Min.   : 1.00  
##  1st Qu.:202   1st Qu.:14.21  
##  Median :403   Median :20.07  
##  Mean   :403   Mean   :18.93  
##  3rd Qu.:604   3rd Qu.:24.58  
##  Max.   :805   Max.   :28.37
```

## Creating the NN Model

The neuralnet library is installed and used to create the model.


```r
library(neuralnet)
```

```
## Warning: package 'neuralnet' was built under R version 4.0.2
```


The model is of the form of $sqrt$~$number$. Different models with varying numbers of hidden layers will be tested for effectiveness. The first model will have 4 hidden layers ,the second will have 16 and the third will have 32.The use of act.fct = logistic for the smoothing of the result. When logistic is used we make the linear.output = FALSE. 


```r
nn<-neuralnet(sqrt~number+sqrt, data=myData,hidden=4,act.fct="logistic",linear.output = FALSE)
nn2<-neuralnet(sqrt~number+sqrt, data=myData,hidden=16,act.fct="logistic",linear.output = FALSE)
nn3<-neuralnet(sqrt~number+sqrt, data=myData,hidden=32,act.fct="logistic",linear.output = FALSE)
```


## Plotting the Results

We can visualize the neural networks:

### Model 1

Plot for the first model:


```r
plot(nn)
```
Neural Net Plot: ![](C:\Users\User\Desktop\SUMMER 2020\APT3010\m1)



### Model 2

Plot for the second model:


```r
plot(nn2)
```
Neural Net Plot: ![](C:\Users\User\Desktop\SUMMER 2020\APT3010\m1)

### Model 3

Plot for the third model:


```r
plot(nn3)
```
Neural Net Plot: ![](C:\Users\User\Desktop\SUMMER 2020\APT3010\m3)


## Creating the Test Dataset

We will use a data set that has numbers ranging from 810 to test the model.


```r
number<-c(810)
testData<- data.frame(number, sqrt=sqrt(number))
testData
```

```
##   number    sqrt
## 1    810 28.4605
```



## Testing the models

We can now test the model using the testing Data Frame created.

### Model 1


```r
predict<-compute(nn,testData)
predict
```

```
## $neurons
## $neurons[[1]]
##        number    sqrt
## [1,] 1    810 28.4605
## 
## $neurons[[2]]
##      [,1] [,2] [,3]         [,4] [,5]
## [1,]    1    1    1 6.871289e-99    1
## 
## 
## $net.result
##           [,1]
## [1,] 0.9999995
```

### Model 2


```r
predict<-compute(nn2,testData)
predict
```

```
## $neurons
## $neurons[[1]]
##        number    sqrt
## [1,] 1    810 28.4605
## 
## $neurons[[2]]
##      [,1] [,2] [,3] [,4] [,5] [,6]         [,7] [,8]          [,9] [,10] [,11]
## [1,]    1    0    1    0    1    1 1.387404e-28    1 7.321626e-158     1     1
##      [,12] [,13] [,14] [,15] [,16] [,17]
## [1,]     0     1     1     1     1     1
## 
## 
## $net.result
##           [,1]
## [1,] 0.9999996
```

### Model 3

```r
predict<-compute(nn3,testData)
predict
```

```
## $neurons
## $neurons[[1]]
##        number    sqrt
## [1,] 1    810 28.4605
## 
## $neurons[[2]]
##      [,1] [,2] [,3] [,4] [,5] [,6] [,7]          [,8]          [,9] [,10] [,11]
## [1,]    1    1    1    1    1    0    1 1.390959e-197 3.181989e-120     1     0
##      [,12] [,13] [,14] [,15] [,16] [,17] [,18] [,19] [,20] [,21] [,22] [,23]
## [1,]     1     1     0     1     1     1     1     0     1     1     1     1
##      [,24]        [,25]        [,26]         [,27] [,28] [,29]        [,30]
## [1,]     1 4.265923e-64 5.714139e-89 4.851386e-224     0     1 2.476602e-43
##              [,31] [,32] [,33]
## [1,] 2.094252e-208     1     1
## 
## 
## $net.result
##      [,1]
## [1,]    1
```


## Conclusion

All three models are able to predict the square roots of 810 up to 7 decimal places.Though, they have different net results.
From the three models created above, the net result increases with an increase in the number of hidden layers. With 32 hidden layers having the greatest value of 0.9999998, proving that it is the most effective model. 
