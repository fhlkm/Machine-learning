data(iris)
########################
## 2-class dataset with continuous 1D features
########################
require(cvTools)
require("e1071")
require(mass)
require(klaR)
require(ROCR)
# get data from R
Iris <- iris[1:100,4:5]
Iris <- droplevels(Iris)
# using linear Discriminant Analysis get model
model <- lda(Species ~ ., data = Iris)
# get prediction
pred  <- predict(model, Iris[,1:2])
yhat=apply(pred$posterior,1,which.max) # argmax
# get confusion matrix
table(yhat, Iris[,2])
#compute error
y    <- as.numeric(Iris[,2]=='setosa')# only one feathure ,the species is column two

# precision-recall curves
pred <- prediction(pred$posterior[,1],y)# for class setosa
perf <- performance(pred, "prec","rec")
plot(perf, xlim = c(0,1), ylim = c(0,1))
# f-measure
fmeasure <- performance(pred,"f")
plot(fmeasure)
#AUC
accuracy <- performance(pred,"acc") 
plot(accuracy)

#########################
##cross  validation
########################################
require(cvTools)

set.seed(1234) # set seed for reproducibility
confusionMatrix <- matrix(0,2,2)
k <- 5         # number of folds
folds <- cvFolds(nrow(Iris), K = k, type = "interleaved")

for(i in 1:k){
  testdata  <- subset(Iris, folds$which == i)
  traindata <- subset(Iris, folds$which != i)
  
  model <- lda(Species ~ ., data = Iris)
  pred  <- predict(model, Iris[,1:2])
  yhat=apply(pred$posterior,1,which.max) # argmax
  
  confusionMatrix <- confusionMatrix + table(yhat,Iris[,2])
}
confusionMatrix/5
###########################################################

