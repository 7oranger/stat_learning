# -*- coding:utf-8 -*-
'''
Created on 2016-10-04
stat leaning Ex2: chapter 2 programming 3.
Plot boxes to show the distributioins of the 4 variables in the Iris data set
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
mean_x=np.array(np.mean(x, axis=0)) # column
#mean_y=np.mean(y)
print mean_x
print mean_x.shape

mean_x_mat=np.zeros(size)
print mean_x_mat.shape
#for i in range(len(y)):
#    mean_x_mat[i,:]=mean_x
mean_x_mat=np.tile(mean_x,(len(y), 1))
#t=mean_x_mat_t-mean_x_mat
#print "diff",t
print mean_x_mat.shape
print np.mean(mean_x_mat,axis=0)
scaled_x=x-mean_x_mat
#scaled_y=y-mean_y

''' 
fig = plt.figure(figsize=(8,6))
 
plt.boxplot(scaled_x,
            notch=False,  # box instead of notch shape
            sym='rs',     # red squares for outliers
            vert=True)   # horizontal box aligmnent
plt.xticks([1,2,3,4],['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.xlabel('data types)
plt.ylabel('scaled values,cm')
t = plt.title('Box plot of Iris data')
plt.show()
'''
f=plt.figure(figsize=(12,5))#创建图表1  

ax1=plt.subplot(121)
ax2=plt.subplot(122)

plt.sca(ax1)  
plt.boxplot(scaled_x,
            notch=False,  # box instead of notch shape
            sym='rs',     # red squares for outliers
            vert=True)   # horizontal box aligmnent
plt.xticks([y+1 for y in range(b)], ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.xlabel('data types')
plt.ylabel('scaled values,cm')
plt.title('Box plot of scaled Iris data')
plt.sca(ax2)  
plt.boxplot(x,
            notch=False,  # box instead of notch shape
            sym='rs',     # red squares for outliers
            vert=True)   # horizontal box aligmnent

plt.xticks([y+1 for y in range(b)], ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.xlabel('data types')
plt.ylabel('original values,cm')
plt.title('Box plot of Iris data')
plt.savefig('box plot of iris data.png')
plt.show(f)  

