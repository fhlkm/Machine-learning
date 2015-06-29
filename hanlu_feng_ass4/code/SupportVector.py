import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets



def getdataset():
    X1=[]
    X2=[]
    Y1=[]
    Y2=[]
    for x in range(0,20):
        for y in range(0,20):
            X1.append(x*0.1)
    for y in range (0,20):
        for x in range(0,20):

            Y1.append(x*0.1)

    X2=[]

    Y2=[]
    for x in range(30,50):
        for y in range(0,20):
            X2.append(x*0.1)
    for y in range (0,20):
        for x in range(30,50):

            Y2.append(x*0.1)
    return X1,Y1,X2,Y2


# dataX1= dataX1[:100]
# dataX2= dataX2[:100]
# dataY1= dataY1[:100]
# dataY2 = dataY2[:100]
def plotOriginal(dataX1,dataY1,dataX2,dataY2):

    plt.title("Test")
    plt.scatter(dataX1,dataY1,c='r')
    plt.scatter(dataX2,dataY2,c='b')
    # plt.show()
    # plt.clf()
    # change origianl data to can calcuate it by svm
def processOriginal(dataX1,dataY1,dataX2,dataY2):
    X1,Y1=    getPartition(dataX1,dataY1,0)
    X2,Y2=    getPartition(dataX2,dataY2,1)
    X = X1+X2
    y= Y1+Y2
    X = np.array(X)
    y = np.array(y)
    return X,y

def getPartition(xdata,ydata, classname):
    dataPart=[];
    classPart=[];
    for i in range(0, len(xdata)):
        group=[xdata[i],ydata[i]]
        dataPart.append(group)
        classPart.append(classname)
    return dataPart,classPart


# print X[:,0]
# X= tuple(X)

# option ,o is svm,  1 is polynomial,2 is guassian kernel,
def plotSupportVector(X,y, option):

    h = .02  # step size in the mesh
    C = 1.0  # SVM regularization parameter
    # clf = svm.SVC(kernel='linear', C=C).fit(X, y)
    clf=0
    if(option ==0):
        plt.title("HardMargin-Linear")
        clf= svm.SVC(kernel='linear', C=C)
    if(option == 2):#gaussian
        plt.title("Gaussian")
        clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
    if(option ==1):# poly
        plt.title("Poly")
        clf = svm.SVC(kernel='poly', degree=3, C=C)
    clf = clf.fit(X, y)

    vectors = clf.support_vectors_
    plt.scatter(vectors[:,0],vectors[:,1],c='y')
    plt.show()
    plt.clf()


def plotSVM(dataX1,dataY1,dataX2,dataY2,option):
    X,y = processOriginal(dataX1,dataY1,dataX2,dataY2)
    plotSupportVector(X,y,option)
#equal vectors, using different svm to calcualte and get the support vectors
dataX1,dataY1,dataX2,dataY2= getdataset()
plotOriginal(dataX1,dataY1,dataX2,dataY2)
plotSVM(dataX1,dataY1,dataX2,dataY2,0)
plotOriginal(dataX1,dataY1,dataX2,dataY2)
plotSVM(dataX1,dataY1,dataX2,dataY2,1)
plotOriginal(dataX1,dataY1,dataX2,dataY2)
plotSVM(dataX1,dataY1,dataX2,dataY2,2)


def printMethod(dataX1,dataY1,dataX2,dataY2):
    print "separated"
    for i in range(0,len(dataX1)):
        print dataX2[i], "    ",dataY2[i],"    ","2", "\n"


# printMethod(dataX1,dataY1,dataX2,dataY2)

def getdataConset():
    X1=[]
    X2=[]
    Y1=[]
    Y2=[]
    for x in range(0,10):
        for y in range(0,10):
            X1.append(x*0.9)
    for y in range (0,10):
        for x in range(0,10):

            Y1.append(x*0.9)

    X2=[]

    Y2=[]
    for x in range(22,42):
        for y in range(0,20):
            X2.append(x*0.3)
    for y in range (0,20):
        for x in range(22,42):
            Y2.append(x*0.3)
    return X1,Y1,X2,Y2

def processOriginal2(dataX1,dataY1,dataX2,dataY2):
    X1,Y1=    getPartition(dataX1,dataY1,0)
    X2,Y2=    getPartition(dataX2,dataY2,1)
    X = X1+X2
    y= Y1+Y2
    X = np.matrix(X)
    y = np.array(y)
    return X,y
# one class have more vectors using linear method to calcualte and get support vectors
dataX1,dataY1,dataX2,dataY2= getdataConset()
def printMethod2(dataX1,dataY1,dataX2,dataY2):
    print "separated"
    for i in range(0,len(dataX1)):
        print dataX2[i], "    ",dataY2[i],"    ","2"
        print dataX1[i], "    ",dataY1[i],"    ","1"
# printMethod2(dataX1,dataY1,dataX2,dataY2)
plotOriginal(dataX1,dataY1,dataX2,dataY2)
plotSVM(dataX1,dataY1,dataX2,dataY2,0)

dataX1,dataY1,dataX2,dataY2= getdataset()
plotOriginal(dataX1,dataY1,dataX2,dataY2)
X,y = processOriginal2(dataX1,dataY1,dataX2,dataY2)
plotOriginal(dataX1,dataY1,dataX2,dataY2)
import cvxopt
import svmpy
import numpy as np
import matplotlib.pyplot as plt
num_samples=500
num_features=2
# X= np.reshape(X,(500,2.0))
# y = np.reshape(y,(500,1.0))
samples = np.matrix(np.random.normal(size=num_samples * num_features)
                        .reshape(num_samples, num_features))
labels = 2 * (samples.sum(axis=1) > 0) - 1.0
print labels[1]
for i in range(0,len(labels)):
    labels[i]=y[i]
print labels[1]
trainer = svmpy.SVMTrainer(svmpy.Kernel.linear(), 0.1)
#
predictor = trainer.train(samples, labels)
vectors = predictor._support_vectors# get the vector of softmargin
plt.scatter(vectors[:,0],vectors[:,1],c='y')
plt.title("Softmargin")
plt.show()
plt.clf()
