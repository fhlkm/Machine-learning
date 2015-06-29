import string
import numpy as np
import math
import matplotlib.pyplot as plt

clusterlength =5
class clusterInfo:
    id=0# the index of cluster
    clusterdata=[]# info of cluster
    datainfolist=[]# info of data
    clusterCenter=[]
    knownId =0

    def setKnownID(self,kId):
        self.knownId =kId
    def getKnownId(self):
        return self.knownId
    def setCluster(self,mcluster):
        cluster=mcluster

    def getCluster(self):
        return self.clusterdata

    def setClusterId(self,idn):
        self.id =idn

    def getId(self):
        return self.id

    def getdatainfoList(self):
        return self.datainfolist

    def addatainfoList(self, element):# add the data info into list
        self.datainfolist.append(element)
    def setdatainfolist(self, newlist):
        self.datainfolist= newlist
    def setClusterCenter(self, centercluster):
        self.clusterCenter= centercluster
        # return clusterCenter ,it is possible ,no value,it is empty
    def getClusterCenter(self):
        if(len(self.clusterCenter)==0):
            mlist=()
            for i in range(0,len(self.datainfolist)):
                if(i==0):
                    mlist  =list( self.datainfolist[i])
                else:
                    for k in range(0,len(self.datainfolist[i])):
                        mlist[k]= float(mlist[k])+float(self.datainfolist[i][k])
            if(len(self.datainfolist)>=2):
                for i in range(0,len(mlist)):
                    mlist[i] = mlist[i]/(len(self.datainfolist))


            self.clusterCenter = tuple(mlist)

            return self.clusterCenter


        else:
            return self.clusterCenter


# read data from file
def read_date_from_file(name):
        fp =open(name);
        dataX =[]
        dataY=[]
        for line in fp.readlines():
            temp = line.strip('\n')
            row = temp.split(',')
            row2=[]
            row2.append(row[0])
            row2.append(row[1])
            row = row2
            length = len(row)
            dimension = length
            row = tuple(list(row))
            # y = row[length-1].strip('\r')#get last item
            # row = row[0:length-1]

            dataX.append(row)
            # dataY.append(y)

        return dataX, dimension

cluster=[]# a list with the centers of clusters
dimension=0
# initiliaze cluster
# def iniCluster():
#     for i in range(0,clusterlength):
#         clustervalue =[]
#         for j in range(0,dimension):
#             clustervalue.append(i*j)
#         cluster.append(clustervalue)
#     return cluster

def iniCluster(data):
    for i in range(0,clusterlength):
        clustervalue =[]
        if (i==0):
            for value in data[i]:
                clustervalue.append(value)
        else:
            index = len(data)/i
            for value in data[int(index-1)]:
                clustervalue.append(value)
        cluster.append(clustervalue)

        #
        # for j in range(0,dimension):
        #     clustervalue.append(i*j)
        # cluster.append(clustervalue)
    return cluster
# get the list of cluster info
def initlaClusterInfolist(clusters):
        listclusterInfo=[]# a list contain all of the cluster info and datainfo
        for i in range(0,len(clusters)):
            clusterinfo = clusterInfo()
            clusterinfo.setClusterId(i)
            clusterinfo.setCluster(cluster[i])
            datainfo =[]
            clusterinfo.setdatainfolist(datainfo)
            clusterCenter =[]
            clusterinfo.setClusterCenter(clusterCenter)
            listclusterInfo.append(clusterinfo)
        return listclusterInfo
# return the listClusterInfo that contain the input data with labels
def getlabel(data,clusters, listClusterInfo):


    for d in data:
            distancelist=[]
            for cluster in clusters:
                if(len(cluster)>0):# if it is not null
                    sum =0
                    for r in range(0,len(cluster)):

                        sum = pow(float(d[r])-float(cluster[r]),2)
                    distancelist.append(sum)
            index = getSmallistClusterIndex(distancelist)
            clusterInfo = listClusterInfo[index]
            clusterInfo.addatainfoList(d)
    return listClusterInfo


# return the smallest index in distancelist
def getSmallistClusterIndex(distancelsit):
    smallest =distancelsit[0]
    smallIndex =0
    for i in range(0,len(distancelsit)):
        if(distancelsit[i]<smallest):
            smallIndex = i
    return smallIndex
## draw "customersdata.csv"
data, dimension= read_date_from_file("customersdata.csv")
cluster = iniCluster(data)# first step initilize the cluster the number is cluster length
bradNclistInfolist= initlaClusterInfolist(cluster)# a brand new clusterInfolist on ly the number equeal the clusters' numer
listclistInfo = getlabel(data,cluster,bradNclistInfolist)# get cluster list with its data and cluster
cluster=[]
for i in range(0,len(listclistInfo)):
    cluster.append(listclistInfo[i].getClusterCenter())

miterative =20

def meprocess(cluster,data, iterative):
        bradNclistInfolist= initlaClusterInfolist(cluster)
        iterative= iterative-1
        listclistInfo = getlabel(data,cluster,bradNclistInfolist)# get cluster list with its data and cluster
        cluster=[]
        for i in range(0,len(listclistInfo)):# e step
             cluster.append(listclistInfo[i].getClusterCenter())
        if (iterative<=0):
            return listclistInfo
        return meprocess(cluster,data,iterative)

m = meprocess(cluster,data, miterative)
# cluster=[]
# for i in range(0,len(listclistInfo)):
#     cluster.append(listclistInfo[i].getClusterCenter())
plt.title("K-means")

for k in range(0,len(m)):
        clu = m[k]
        datalist = clu.getdatainfoList()
        colovrValue=['r','b','y','c','m','g']
        for i in range(0,len(datalist)):
            value = datalist[i]
            plt.plot(value[0],value[1],'.',c=colovrValue[k])
            center = clu.getClusterCenter()
            plt.plot(center[0],center[1],'s',c=colovrValue[k])
plt.show()
plt.clf()

####################draw D_spatial_network.txt

data, dimension= read_date_from_file("D_spatial_network.txt")
cluster = iniCluster(data)# first step initilize the cluster the number is cluster length
bradNclistInfolist= initlaClusterInfolist(cluster)# a brand new clusterInfolist on ly the number equeal the clusters' numer
listclistInfo = getlabel(data,cluster,bradNclistInfolist)# get cluster list with its data and cluster
cluster=[]
for i in range(0,len(listclistInfo)):
    cluster.append(listclistInfo[i].getClusterCenter())

miterative =20

m = meprocess(cluster,data, miterative)
# cluster=[]
# for i in range(0,len(listclistInfo)):
#     cluster.append(listclistInfo[i].getClusterCenter())
plt.title("K-means")

for k in range(0,len(m)):
        clu = m[k]
        datalist = clu.getdatainfoList()
        colovrValue=['r','b','y','c','m','g']
        for i in range(0,len(datalist)):
            value = datalist[i]
            plt.plot(value[0],value[1],'.',c=colovrValue[k])
            center = clu.getClusterCenter()
            plt.plot(center[0],center[1],'s',c=colovrValue[k])
plt.show()
plt.clf()



#
# print (5)
#####################################Em Algorithm#####################################
# read data from file
# read data from file
def read_date_from_file2(name):
        fp =open(name);
        dataX =[]
        dataY=[]
        for line in fp.readlines():
            temp = line.strip('\n')
            row = temp.split(',')
            length = len(row)
            dimension = length
            row2=[]
            for v in row:
                row2.append(float(v))
            # y = row[length-1].strip('\r')#get last item
            # row = row[0:length-1]

            dataX.append(row2)
            # dataY.append(y)

        return dataX, dimension
data, dimension= read_date_from_file2("customersdata.csv")
sum_abs =1

# get square matrix the lenth and column equal the row
def initialMatrix(row):
    matrix =[]
    for j in range(0,row):
        rows =[]
        for i in range(0,row):
            rows.append(1)
        matrix.append(rows)
    matrix =np.matrix(matrix)
    return matrix

def initialSMatrix(row, column):
    matrix =[]
    for j in range(0,row):
        rows =[]
        for i in range(0,column):
            rows.append(0)
        matrix.append(rows)
    matrix =np.matrix(matrix)
    return matrix
def initialUMatrix(column):
    rows =[]
    for i in range(0,column):
            rows.append(0)
    return np.matrix(rows)

k =3
sum_matrix = initialMatrix(len(data))
# sum_matrix = initialSMatrix(len(data),dimension)
det_Matrix = np.linalg.det(sum_matrix)+1# sum's det
data_matrix = np.matrix(data)
u_matrix = initialUMatrix(dimension)
mm= np.subtract(np.matrix(np.array(data_matrix[0,:])[0]),np.matrix(np.array(u_matrix[0,:])[0]))
exp_value=(-1/2)*det_Matrix*np.subtract(np.matrix(np.array(data_matrix[0,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])).dot(np.subtract(np.matrix(np.array(data_matrix[0,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])).T)
exp_value = np.array(exp_value[0,:])[0][0]
p_l_i = pow(math.e,exp_value)/(pow(2*math.pi,1/2)*det_Matrix)
alpha_l_guss =1/k;
def getsum_denominator(data):#only get when i =0
    sum =0
    for i in range(0,k):
        # for j in range(0, len(data) ):# we shoul change _theta_j and sum_det_j
            exp_value=(-1/2)*det_Matrix*np.subtract(np.matrix(np.array(data_matrix[0,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])).T.dot(np.subtract(np.matrix(np.array(data_matrix[0,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])))
            exp_value = np.array(exp_value[0,:])[0][0]
            p_j_i = pow(math.e,exp_value)/(pow(2*math.pi,1/2)*det_Matrix)
            p_j_i=alpha_l_guss*p_l_i
            sum = sum+p_j_i
    return sum

p_l_i_g = alpha_l_guss*p_l_i/getsum_denominator(data)
def getSumP():
    sum =0
    for i in range(0, len(data)):
        exp_value=(-1/2)*det_Matrix*np.subtract(np.matrix(np.array(data_matrix[i,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])).dot(np.subtract(np.matrix(np.array(data_matrix[i,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])).T)
        exp_value = np.array(exp_value[0,:])[0][0]
        p_l_i = pow(math.e,exp_value)/(pow(2*math.pi,1/2)*det_Matrix)
        sum = sum+p_l_i
    sum = sum/len(data)
    return sum

alpah_new = (1/len(data))*getSumP()

def calculateData(filename):
    data, dimension= read_date_from_file(filename)
    cluster = iniCluster(data)# first step initilize the cluster the number is cluster length
    bradNclistInfolist= initlaClusterInfolist(cluster)# a brand new clusterInfolist on ly the number equeal the clusters' numer
    listclistInfo = getlabel(data,cluster,bradNclistInfolist)# get cluster list with its data and cluster
    cluster=[]
    for i in range(0,len(listclistInfo)):
        cluster.append(listclistInfo[i].getClusterCenter())

    miterative =10

    def meprocess(cluster,data, iterative):
            bradNclistInfolist= initlaClusterInfolist(cluster)
            iterative= iterative-1
            listclistInfo = getlabel(data,cluster,bradNclistInfolist)# get cluster list with its data and cluster
            cluster=[]
            for i in range(0,len(listclistInfo)):# e step
                 cluster.append(listclistInfo[i].getClusterCenter())
            if (iterative<=0):
                return listclistInfo
            return meprocess(cluster,data,iterative)

    m = meprocess(cluster,data, miterative)
    # cluster=[]
    # for i in range(0,len(listclistInfo)):
    #     cluster.append(listclistInfo[i].getClusterCenter())
    plt.title("Expection-Maximization")

    for k in range(0,len(m)):
            clu = m[k]
            datalist = clu.getdatainfoList()
            colovrValue=['r','b','y','c','m','g']
            for i in range(0,len(datalist)):
                value = datalist[i]
                plt.plot(value[0],value[1],'.',c=colovrValue[k])
                center = clu.getClusterCenter()

                plt.plot(center[0],center[1],'s',c=colovrValue[k])
    plt.show()


def getSumPVector():
    u_matrix = initialUMatrix(dimension)
    for i in range(0, len(data)):
        exp_value=(-1/2)*det_Matrix*np.subtract(np.matrix(np.array(data_matrix[i,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])).dot(np.subtract(np.matrix(np.array(data_matrix[i,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])).T)
        exp_value = np.array(exp_value[0,:])[0][0]
        p_l_i = pow(math.e,exp_value)/(pow(2*math.pi,1/2)*det_Matrix)
        p_l_i*data_matrix[i,:]
        u_matrix = np.add(u_matrix,p_l_i)
    u_matrix = u_matrix/len(data)
    return u_matrix
u_new = (1/(len(data)*alpah_new))*getSumPVector()
def getSigmaSumPVector():
    u_matrix = initialUMatrix(dimension)
    for i in range(0, len(data)):
        exp_value=(-1/2)*det_Matrix*np.subtract(np.matrix(np.array(data_matrix[i,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])).dot(np.subtract(np.matrix(np.array(data_matrix[i,:])[0]),np.matrix(np.array(u_matrix[0,:])[0])).T)
        exp_value = np.array(exp_value[0,:])[0][0]
        p_l_i = pow(math.e,exp_value)/(pow(2*math.pi,1/2)*det_Matrix)
        p_l_i*data_matrix[i,:]
        u_matrix = np.add(u_matrix,p_l_i)
    u_matrix = u_matrix/len(data)
    return u_matrix
sign_new= (1/(len(data)*alpah_new))*getSigmaSumPVector()


calculateData("customersdata.csv")
calculateData("D_spatial_network.txt")




