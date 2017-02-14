# -*- coding: utf-8 -*-
'''
Created on 2016��12��3��

@author: RenaiC
'''
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np

w = [0,0,0]  # for alpha
b = 0 # for bias
yita = 1   #learning rate
N = 3 
data = [ [(3,3),1],  [(4,3),1],  [(1,1),-1]  ]

data_x = np.array( [data[k][0] for k in xrange(N) ])
print data_x
data_y = [data[k][1] for k in xrange(N) ]
print data_y
result = []  #trace for w
# 3 samples
# gram_mat = np.zeros([N,N])
# for i in xrange(N):
#     for j in xrange(N):
#         gram_mat[i,j]= np.dot(data_x[i,:],data_x[j,:])
def predict(x):
    t = 0
    for i in xrange(N):
        t = t + w[i]*data_y[i]*np.dot(data_x[i,:],x)
    y = t+b
    if y>0:
        y_p = 1
    else:
        y_p = -1
    return y_p

def update(x,i):
    'update if the i_th sample is wrongly classified'
    global w,b,record
    w[i] = w[i] + yita 
    b = b + yita* data_y[i]
    result.append([w,b])
    print (w,b)
#     print result
    
def perceptron():
    'one turn'
    count = 1
    for ii in xrange(N):
        item = data_x[ii,:]
        y = predict(item)
        if y == data_y[ii]:
            'right'
            count = count +1
        else:
            'wrong,then update'
            count = 1
            update(item,ii)
    if count >= len (data):
        'all samples are classified correctly'
        return 1
    else:
        'some samples goes wrong'
        return 0
         
if __name__ == "__main__":
    while 1:
        a = perceptron()
        if a > 0:
            break
    w_final=result[-1]
    print 'Final parameter:',result[-1]
#     #draw a figure to demostrate
#     plt.figure()
#     plt.title('perceptron  demo')
#     for item in data:
#         if item[1]==1:
#             plt.scatter(item[0][0],item[0][1],marker = 'x', color = 'y')
#         else:
#             plt.scatter(item[0][0],item[0][1],marker = 'o', color = 'b')
#     
#     x = range(-3,6)
#     if w[1] == 0:
#         t = -1*w[2]/w[0]
#         x = [t]*6
#         y = xrange(6)
#     else:
#         y = [ (-1*w[2]-w[0]*t)/w[1] for t in x]
#     plt.plot(x,y,'r')
#     plt.savefig('perception-origin.png')  
#     plt.show()
#     print 'record\n', result

