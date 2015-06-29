require(cvTools)
require("e1071")
require(mass)
require(klaR)
require(ROCR)
########################
## n-class dataset with  k class  and continuous nD features
########################
Iris <- iris[1:150,1:5]
Iris <- droplevels(Iris)
model <- lda(Species ~ ., data = Iris)
pred  <- predict(model, Iris[,1:4])
yhat=apply(pred$posterior,1,which.max) # argmax
table(yhat, Iris[,5])

#########################
##cross  validation
########################################

set.seed(1234) # set seed for reproducibility
confusionMatrix <- matrix(0,3,3)
k <- 5         # number of folds
folds <- cvFolds(nrow(Iris), K = k, type = "interleaved")

for(i in 1:k){
  testdata  <- subset(Iris, folds$which == i)
  traindata <- subset(Iris, folds$which != i)
  
  model <- lda(Species ~ ., data = Iris)
  pred  <- predict(model, Iris[,1:5])
  yhat=apply(pred$posterior,1,which.max) # argmax
  
  confusionMatrix <- confusionMatrix + table(yhat,Iris[,5])
}
confusionMatrix/5