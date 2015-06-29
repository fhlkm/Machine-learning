import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg



def draw_pcitures(name):
        data = np.loadtxt(name)
        x = data[:,0]
        y = data[:,1]
        plt.title(name)
        plt.scatter(x,y,c='r')
        plt.show()
        plt.clf()

def read_date_from_file(name, fold):
        data = np.loadtxt(name)
        x = data[:,0]
        y = data[:,1]

        # plt.scatter(x, y)
        # plt.show()

        size = len(x);
        # print "date size is %n \n", size;

        train_x=x[0:size/10*fold]
        test_x = x[size/10*9:size]# the last ten percent as test date
        # print "training date size is %n\n", len(test_x)


        train_y= y[0:size/10*fold]
        # print "test y is\n"
        # print test_y
        test_y= y[size/10*9:size]
        # plt.title(name)
        plt.scatter(x,y,c='b')

        # plt.show()
        # plt.clf()
        return train_x,test_x,train_y,test_y,x


# give x and order we get the matrix the row is the example's order,
def get_Z(data,order):
    arr =[]

    for i in data:# is all of the x
        row_array = create_row(i,order)
        arr.append(row_array)

    return arr

# give x ,we get the row
def create_row(x_value,order):
    a=[]

    for i in range(order+1):
        a.append(x_value ** i)
    return a


# print "Matrix is :\n"
# print matrix

def get_theta(z_matrix,y):
    theta = (z_matrix.T.dot(z_matrix)).I.dot(z_matrix.T)
    theta = theta.dot(y.T)# y is n*1 matrix
    return theta

def get_test_error(x,y,theta,order,filename):
     matrix_data = get_Z(x,order)
     matrix_z = np.matrix(matrix_data)
     cal_y= matrix_z.dot(theta.T)

     arr_cal_y= np.squeeze(np.asarray(cal_y.T))
     plt.title(filename)

     plt.scatter(x, y,c='b')
     plt.scatter(x,arr_cal_y,c='r')
     plt.show()


     cal_matrix_y = np.matrix(y).T
     error_y = np.subtract(cal_y,cal_matrix_y)
     error_matrix_y = np.matrix(error_y)

     sum_test =0.0
     for i  in error_matrix_y:
        sum_test = sum_test +i**2

     sum_test = sum_test/len(y)
     return sum_test




#######calcualte the data############################

def regression_algorithm(filename,fold, order):
    train_x,test_x,train_y,test_y,all_x = read_date_from_file(filename,fold)
    matrix_data = get_Z(train_x,order)
    # we get the matrix
    matrix_z = np.matrix(matrix_data)


    theta = get_theta(matrix_z,train_y)
    print "for %s theta is   " % filename
    print theta
    # graph(all_x,theta,filename,order)
    sum = get_test_error(train_x,train_y,theta,order,filename)
    print ("training Variance is  ");
    print(sum)
    sum = get_test_error(test_x,test_y,theta,order,filename)
    print ("testing Variance is  ");
    sum = np.squeeze(np.asarray(sum))
    print(sum)

def graph(x_range,parameters,filename, order):
      sorted_arr = sorted(x_range)
      first = sorted_arr[0]
      last = sorted_arr[len(sorted_arr)-1]
      rt=range(int(first),int(last))
      x = np.array(rt)
      y = my_formula(parameters,x)
      plt.title("file %s and order %n" %(filename,order))
      plt.plot(x,y)
      plt.show()
      plt.clf()
def my_formula(arr_p,x):

    k =0
    arr = np.squeeze(np.asarray(arr_p))
    for i in range(len(arr)):
        if(i==0):
          k = k+arr[i]
        else:
          k=k+arr[i]*(x**i)
    return k

# draw_pcitures('svar-set1.txt')
# draw_pcitures('svar-set2.txt')
# draw_pcitures('svar-set3.txt')
# draw_pcitures('svar-set4.txt')
print "for 10 fold validation evaluation,order is 3: "
regression_algorithm('svar-set1.txt',9, 3)
regression_algorithm('svar-set2.txt',9, 3)
regression_algorithm('svar-set3.txt',9, 3)
regression_algorithm('svar-set4.txt',9, 3)

#
# print "Using 80 percent of tainging data,order is 3: "
regression_algorithm('svar-set1.txt',8, 3)
regression_algorithm('svar-set2.txt',8, 3)
regression_algorithm('svar-set3.txt',8, 3)
regression_algorithm('svar-set4.txt',8, 3)
print "Using 70 percent of tainging data,order is 3: "
regression_algorithm('svar-set1.txt',7, 3)
regression_algorithm('svar-set2.txt',7, 3)
regression_algorithm('svar-set3.txt',7, 3)
regression_algorithm('svar-set4.txt',7, 3)


# print "for 10 fold validation evaluation,order is 4: "
regression_algorithm('svar-set1.txt',9, 4)
regression_algorithm('svar-set2.txt',9, 4)
regression_algorithm('svar-set3.txt',9, 4)
regression_algorithm('svar-set4.txt',9, 4)
#
# print "for 10 fold validation evaluation,order is 2: "
regression_algorithm('svar-set1.txt',9, 2)
regression_algorithm('svar-set2.txt',9, 2)
regression_algorithm('svar-set3.txt',9, 2)
regression_algorithm('svar-set4.txt',9, 2)

# print "for svar-set2.txt fold validation evaluation,order is 8: "
regression_algorithm('svar-set2.txt',9, 8)
# print "for svar-set2.txt fold validation evaluation,order is 10: "
regression_algorithm('svar-set2.txt',9, 10)
# print "for svar-set2.txt fold validation evaluation,order is 13: "
regression_algorithm('svar-set2.txt',9, 13)
# print "for svar-set2.txt fold validation evaluation,order is 15: "
regression_algorithm('svar-set2.txt',9, 15)




