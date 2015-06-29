########################
## Naive Byes with Binomioal features
########################

require("e1071")
require(ROCR)
require(cvTools)
Toe <- read.table("F:/TOE.data",sep=",")# 23 columns
Toe <- droplevels(Toe)
y    <- as.numeric(Toe[,10]=='positive')

# compute and apply model with raw output 
model<-naiveBayes(Toe[,1:9], Toe[,10]) 
yhat = predict(model, Toe[,-10])
table(yhat,Toe[,10])
# precision-recall curves
yhat = predict(model, Toe[,-10],type="raw")
pred <- prediction(yhat[,1],y)
perf <- performance(pred, "prec","rec")
plot(perf, xlim = c(0,1), ylim = c(0,1))

# fmeasure
fmeasure <- performance(pred,"f")
plot(fmeasure)

# accuracy
accuracy <- performance(pred,"acc") 
plot(accuracy)

#########################
##cross  validation
########################################

set.seed(1234) # set seed for reproducibility
confusionMatrix <- matrix(0,2,2)
k <- 5         # number of folds
folds <- cvFolds(nrow(Toe), K = k, type = "interleaved")

for(i in 1:k){
  testdata  <- subset(Toe, folds$which == i)
  traindata <- subset(Toe, folds$which != i)
  
  model<-naiveBayes(traindata[,1:9], traindata[,10]) 
  yhat = predict(model, testdata[,-10])
  
  confusionMatrix <- confusionMatrix + table(yhat,testdata[,10])
}
confusionMatrix
