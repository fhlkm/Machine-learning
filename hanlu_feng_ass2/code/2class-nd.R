########################
## 2-class dataset with  2 class and continuous nD features
########################
require(cvTools)
require("e1071")
require(mass)
require(klaR)
require(ROCR)
Iris <- iris[1:100,3:5]
Iris <- droplevels(Iris)
model <- lda(Species ~ ., data = Iris)
pred  <- predict(model, Iris[,1:3])
yhat=apply(pred$posterior,1,which.max) # argmax
table(yhat, Iris[,3])

require(cvTools)


#########################
##cross  validation
########################################

set.seed(1234) # set seed for reproducibility
confusionMatrix <- matrix(0,2,2)
k <- 5         # number of folds
folds <- cvFolds(nrow(Iris), K = k, type = "interleaved")

for(i in 1:k){
  testdata  <- subset(Iris, folds$which == i)
  traindata <- subset(Iris, folds$which != i)
  
  model <- lda(Species ~ ., data = Iris)
  pred  <- predict(model, traindata[,1:3])
  yhat=apply(pred$posterior,1,which.max) # argmax
  
  confusionMatrix <- confusionMatrix + table(yhat,Iris[,3])
}
confusionMatrix/5