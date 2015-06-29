from sklearn.linear_model import LogisticRegression
import  numpy as np
from sklearn.cross_validation import KFold


def read_date_from_file(name):
        fp =open(name);
        dataX =[]
        dataY=[]
        for line in fp.readlines():
            temp = line.strip('\n')
            row = temp.split(',')
            length = len(row)
            row = tuple(list(row))
            y = row[length-1].strip('\r')#get last item
            row = row[0:length-1]

            dataX.append(row)
            dataY.append(y)

        return dataX,dataY;

def accuracy(truth, predicted):
    return (1. * len([1 for tr, pr in zip(truth, predicted) if tr == pr]) / len(truth))
# x is all of the data, y all of the data
def crossV(X,y,model):
    cv = KFold(len(y), 5)
    accuracies = []
    for train_ind, test_ind in cv:
        model.fit(X[train_ind], y[train_ind])
        predictions = model.predict(X[test_ind])
        accuracies.append(accuracy(y[test_ind], predictions))
    print 'Average 5-fold cross validation accuracy=%.2f (std=%.2f)' % (np.mean(accuracies), np.std(accuracies))
def doLogi(name):
    model = LogisticRegression()
    dataX,dataY= read_date_from_file(name)
    length = len(dataX)
    trainX = np.array(dataX[0:length/10*9],dtype=float)
    trainY= np.array(dataY[0:length/10*9],dtype=float)
    testX=np.array(dataX[length/10*9:length],dtype = float)
    testY = np.array(dataY[length/10*9:length],dtype =float)

    model.fit(trainX,trainY)
    predict = model.predict(testX)

    print 'accuracy on testing data=%.3f' %  accuracy(testY,predict)
    crossV(trainX,trainY,model)
print "Two classification:"
doLogi('ballon-digit.txt')
print "non-linear inputs:"
doLogi('abalone.txt')

def read_date_from_file2(name):
        fp =open(name);
        dataX =[]
        dataY=[]
        for line in fp.readlines():
            temp = line.strip('\n')
            temp = temp.strip('\r')
            row = temp.split(' ')
            length = len(row)
            row = tuple(list(row))

            y =row[length-11:length-1]
            row = row[0:length-11]
            y = ''.join(y)
            y = (y,)


            dataX.append(row)
            dataY.append(y)

        return dataX,dataY;


def doLogi2(name):
    model = LogisticRegression()
    dataX,dataY= read_date_from_file2(name)
    length = len(dataX)
    trainX = np.array(dataX[0:length/10*9],dtype=float)
    trainY= np.array(dataY[0:length/10*9],dtype=float)
    testX=np.array(dataX[length/10*9:length],dtype = float)
    testY = np.array(dataY[length/10*9:length],dtype =float)

    model.fit(trainX,trainY)
    predict = model.predict(testX)

    print 'accuracy on testing data=%.3f' %  accuracy(testY,predict)
    crossV(trainX,trainY,model)
print("image k-classification is:")
doLogi2('semeion.txt')





