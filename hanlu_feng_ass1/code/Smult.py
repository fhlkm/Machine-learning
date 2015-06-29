import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm
import time


def read_date_from_file(name, fold):
        data = np.loadtxt(name)
        x_one = data[:,0]
        x_two = data[:,1]
        y = data[:,2]

        draw_surface(x_one,x_two,y)


        size = len(x_one);
        # print "date size is %d \n" % size;

        train_x_one=x_one[0:size/10*fold]
        test_x_one = x_one[size/10*9:size]# the last ten percent as test date

        train_x_two=x_two[0:size/10*fold]
        test_x_two = x_two[size/10*9:size]# the last ten percent as test date
        # print "training date size is %n\n", len(test_x)


        train_y= y[0:size/10*fold]
        # print "test y is\n"
        # print test_y
        test_y= y[size/10*9:size]
        return train_x_one,test_x_one,train_x_two,test_x_two,train_y,test_y


def  draw_surface(x,y,z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))

    X, Y = np.meshgrid(xi, yi)
    Z = griddata(x, y, z, xi, yi)

    surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.jet,
                           linewidth=1, antialiased=True)

    ax.set_zlim3d(np.min(Z), np.max(Z))
    fig.colorbar(surf)

    plt.show()
def get_rwo_order2(xone, xtwo):
    arr=[]
    arr.append(1)
    arr.append(xone)
    arr.append(xtwo)
    arr.append(xone*xtwo)
    arr.append(xone**2)
    arr.append(xtwo**2)
    return arr;

def get_z_order2(x_one,x_two):
    arr =[]
    for index in range(len(x_one)):
        xone = x_one[index]
        xtwo = x_two[index]
        row = get_rwo_order2(xone,xtwo)
        arr.append(row)
    return arr

def get_theta(z_matrix,y):
    theta = (z_matrix.T.dot(z_matrix)).I.dot(z_matrix.T)
    theta = theta.dot(y.T)# y is n*1 matrix
    return theta

def get_test_error(xone,xtwo,y,theta):
    arr_x= get_z_order2(xone,xtwo)
    matrix_z=np.matrix(arr_x)
    cal_y= matrix_z.dot(theta.T)# x matrix * theta, get y
    cal_matrix_y = np.matrix(y).T
    error_y = np.subtract(cal_y,cal_matrix_y)
    error_matrix_y = np.matrix(error_y)
    sum_test =0.0
    for i  in error_matrix_y:
        sum_test = sum_test +i**2

    sum_test = sum_test/len(y)
    return sum_test


def getSum_and_theta(filename):
    print "for file %s"%filename
    train_x_one,test_x_one,train_x_two,test_x_two,train_y,test_y = read_date_from_file(filename, 9)
    arr_x= get_z_order2(train_x_one,train_x_two)
    matrix_z=np.matrix(arr_x)
    theta = get_theta(matrix_z,train_y)
    print "theta is "
    print theta

    sum = get_test_error(train_x_one,train_x_two,train_y,theta)
    sum = np.squeeze(np.asarray(sum))
    print "training main square error  is "
    print sum
    sum = get_test_error(test_x_one,test_x_two,test_y,theta)
    sum = np.squeeze(np.asarray(sum))
    print "testing main square error is "
    print sum

# getSum_and_theta('mvar-set1.txt')
# getSum_and_theta('mvar-set2.txt')
# getSum_and_theta('mvar-set3.txt')
# getSum_and_theta('mvar-set4.txt')

def get_feathure_t(xone,xtwo):
      arr=[]
      arr.append(xone)# the first row is feature one
      arr.append(xtwo)# the second row is feature two;
      return arr

def cal_alpha(filename, xone,xtwo):
    train_x_one,test_x_one,train_x_two,test_x_two,train_y,test_y = read_date_from_file(filename, 9)
    arr_x= get_z_order2(train_x_one,train_x_two)
    matrix_z=np.matrix(arr_x)

    feature = get_feathure_t(train_x_one,train_x_two)
    matrix_feature = np.matrix(feature)
    G=matrix_z.dot(matrix_z.T)
    alpha = G.I.dot(np.matrix(train_y).T)

    feature_give = get_feathure_t(xone,xtwo)# give the x,y that we all test
    matrix_feature_given = np.matrix(feature_give)

    arr_x_give= get_z_order2(xone,xtwo)
    matrix_z_give=np.matrix(arr_x_give)

    y = alpha.T.dot(matrix_feature_given).dot(matrix_z_give)


    return y

# def get_error(y_predit,y_real ):
#     error = np.subtract(y_predit,y_real)
#     sum_test =0.0
#     for i  in error:
#         sum_test = sum_test +i**2
#
#     sum_test = sum_test/len(y)
#     return sum_test

def get_RSE(arr):
    rse=0.0
    for i in arr:
        rse = rse+i**2
    rse = rse/len(arr)
    rse = rse**(0.25*0.25*0.25*0.25)
    return rse



def gaussian_algorithm(filename):
    time1=time.time()
    train_x_one,test_x_one,train_x_two,test_x_two,train_y,test_y = read_date_from_file(filename, 9)
    example = get_example(train_x_one,train_x_two)
    pre_matrix = get_G(example)
    matrix_G= np.matrix(pre_matrix)
    alpha = get_alpha(matrix_G,np.matrix(train_y).T)
    alpha =np.asarray(alpha)
    arr=[]#contains the expect value of the y which is clacluate by x
    for j in range(len(example)):
        exmaple_x = example[j]
        y=0.0
        for i in range(len(alpha)):
           y=y+ alpha[i]*k_function(example[i],exmaple_x)
        arr.append(y)
    error =[]
    for k in range(len(arr)):
        difference = arr[k]-train_y[k]
        error.append(difference)
    rse = get_RSE(error)
    print "length of data is %d" % len(arr)
    print rse
    time2 = time.time()
    distance = time2 - time1
    print distance
    return rse



def get_alpha(g,y):
    alpha =g.I.dot(y)
    return alpha
    # y = cal_alpha(filename,train_x_one,train_x_two)
    # error = get_error(y,train_y)
    # print error
# x_one is example one, x_two is example 2
def k_function(example_one,example_two):
    x_i = np.matrix(example_one).T
    x_x = np.matrix(example_two).T
    k = math.exp(x_x.T.dot(x_x)*0.5)*math.exp(x_i.T.dot(x_x))*math.exp(x_i.T.dot(x_i)*0.5)
    return k
def get_example(x_one,x_two):
    arr=[]
    for i in range(len(x_one)):
        example=[]
        example.append(x_one[i])
        example.append(x_two[i])
        arr.append(example)
    return arr

def get_G(examples):
    arr =[]
    for i in range(len(examples)):
         row = get_G_row(examples[i],examples)
         arr.append(row)

    return arr



def get_G_row(exple,exples):
    arr=[]
    for i in range (len(exples)):
        item = k_function(exple,exples[i])
        arr.append(item)
    return arr


getSum_and_theta('mvar-set1.txt')
getSum_and_theta('mvar-set2.txt')
getSum_and_theta('mvar-set3.txt')
getSum_and_theta('mvar-set4.txt')
print "for Gaussian Kernel method: \n"
gaussian_algorithm('mvar-set1.txt')







