
#################################################################
########################
## Naive Byes with Bernoulli features
########################
require("e1071")
require(cvTools)
require(ROCR)
SPEC <- read.table("F:/SPEC.data",sep=",")# 23 columns
SPEC <- droplevels(SPEC)
y    <- as.numeric(SPEC[,23]=='one')

# compute and apply model with raw output 
model<-naiveBayes(SPEC[,1:22], SPEC[,23]) 
yhat = predict(model, SPEC[,-23])
table(yhat,SPEC[,23])
# precision-recall curves
yhat = predict(model, SPEC[,-23],type="raw")
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
folds <- cvFolds(nrow(SPEC), K = k, type = "interleaved")

for(i in 1:k){
  testdata  <- subset(SPEC, folds$which == i)
  traindata <- subset(SPEC, folds$which != i)
  
  model<-naiveBayes(traindata[,1:22], traindata[,23]) 
  yhat = predict(model, testdata[,-23])
  
  confusionMatrix <- confusionMatrix + table(yhat,testdata[,23])
}
confusionMatrix

