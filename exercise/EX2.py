# -*- coding:utf-8 -*-
'''
Created on 2016年10月4日
stat leaning Ex2: chapter 2 programming 1. KLT
http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
http://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html
@author: RENAIC225
'''
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
label=iris.target
a,b=iris.data.shape
print a,b
x=iris.data
y=iris.target
size=iris.data.shape
print x[1,:]
mean_x=np.array(np.mean(x, axis=0)) #column
#mean_y=np.mean(y)
print mean_x
print mean_x.shape

mean_x_mat=np.zeros(size)
print mean_x_mat.shape
for i in range(len(y)):
    mean_x_mat[i,:]=mean_x
print mean_x_mat.shape
print np.mean(mean_x_mat,axis=0)
scaled_x=x-mean_x_mat
#scaled_y=y-mean_y
cov=np.cov(scaled_x.transpose())
corr=np.corrcoef(scaled_x,rowvar=False )#If rowvar is True (default), then each row represents a variable, with observations in the columns. 
#Otherwise, the relationship is transposed: each column represents a variable, while the rows contain observations.
print cov  #4*4
print corr  #4*4
eigval, eigvec=np.linalg.eig(cov)
print eigvec.shape #4*4
#klt = np.transpose(np.dot(eigvec,scaled_x.transpose()))
klt=np.dot(scaled_x,eigvec)  #same to last line
print klt.shape  # 150*4
'''
def KLT(a):
    """
    Returns Karhunen Loeve Transform of the input and 
    the transformation matrix and eigenval
     
    Ex:
    import numpy as np
    a  = np.array([[1,2,4],[2,3,10]])
     
    kk,m = KLT(a)
    print kk
    print m
     
    # to check, the following should return the original a
    print np.dot(kk.T,m).T
         
    """
    val,vec = np.linalg.eig(np.cov(a))
    klt = np.dot(vec,a)
    return klt,vec,val
'''
