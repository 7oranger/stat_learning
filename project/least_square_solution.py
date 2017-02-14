# -*- coding: utf-8 -*-
from __future__ import division
'''
Created on 2016-12-10

@author: RenaiC
'''
import numpy as np
import os,math,random
import matplotlib.pyplot as plt

def load_data():
    xtrain = np.loadtxt('./data/xtrain.txt', delimiter = "," , usecols=(0,1) , dtype='float');# two column
    ctrain = np.loadtxt('./data/ctrain.txt', delimiter = "," , usecols=(0,) , dtype='int');# one column 
    xtest = np.loadtxt('./data/xtest.txt', delimiter = "," , usecols=(0,1) , dtype='float');# two column
    ptest = np.loadtxt('./data/ptest.txt', delimiter = "," , usecols=(0,) , dtype='float');#
    c1test = np.loadtxt('./data/c1test.txt', delimiter = "," , usecols=(0,) , dtype='float');
    return xtrain,ctrain,xtest,ptest,c1test # 200 个训练样本，6821个测试
def get_phi(x,order):
    'order =  1 2'
    height = len(x)
    a = np.zeros([height,order*2+1])
    for i in xrange(height):
        for j in xrange(order+1):# 0 ~ order
            a[i,j]=math.pow(x[i,0],j)
    for i in xrange(height):
        for j in range(1,order+1):# 1 ~ order
            a[i,j]=math.pow(x[i,1],j)
    return a
def get_w(phi,order,y):
    tmp0 = np.linalg.inv(np.dot(phi.T,phi)+np.eye(2*order+1))
    tmp1 = np.dot(tmp0,phi.T)
    w = np.dot(tmp1, y )
    return w
    #y_p = np.dot(get_phi(x,order),omega)

def get_w_and_error(x,y,order):
    'give x and y, find the error by K fold-cv. error = average(each turn)'
    K = 5
    a = len(x)
    test_num = 1./K * a# K-fold CV 中用于测试的样本个数
    #train_num = a -  test_num # K-fold CV 中用于训练的样本个数
    index = range(a)
    random.shuffle(index)  # 打乱
    x = x[index]
    y = y[index]
    error_kcv = []
    w = []
    for i in xrange(5):
        start = test_num*i # 测试集的起点和终点
        end = test_num*(i+1)
        x_te = x[start:end] #测试
        y_te = y[start:end] #训练
        x_tr = np.vstack( (x[0:start],x[end:]) ) #剩下的就是训练集
        y_tr = np.concatenate( (y[0:start],y[end:]) )#vstack
        xtr_phi = get_phi(x_tr,order)
        w0 =  get_w(xtr_phi,order,y_tr)
        w.append(w0) # 存放所有 w 值
        ### 用于验证集
        y_predict = np.dot(get_phi(x_te,order),w0)#np.transpose(w0)
        error_kcv.append(compare(y_predict,y_te))
    wy = get_w(get_phi(x,order),order,y)
    error = np.mean(error_kcv)
    return wy, error

def compare(y1,y2):
    'compare the sign between y1 and y2'
    count = 0
    a = len(y1)
    for i in xrange(a):
#         print 'y_predict' ,y1[i],'y_te',y2[i]
        if np.sign(y1[i]) != np.sign(y2[i]) :
            count = count +1
    return count*1.0/a

def get_eval_error(w,xtest,ptest,c1test,order):
    test_phi = get_phi(xtest,order)   
    predicted_y = np.dot(test_phi,w)
    error = 0
    for i in xrange(len(predicted_y)):
        if np.sign( predicted_y[i] ) == -1: # 被分为负例
            error = error + ptest[i]*c1test[i]
        else:
            error = error + ptest[i]*(1-c1test[i])
    return error 

def raw_ls_regression():
    xtrain,ctrain,xtest,ptest,c1test = load_data()
    print ctrain.shape
    for i in xrange(len(ctrain)):
        if ctrain[i] == 0:
            ctrain[i] = -1
    order_list = range(1,10) # 1 ~ 9
    error0 = []
    error1 = []
    for order in order_list:
        w, err = get_w_and_error( xtrain,ctrain,order)
        error0.append(err)
        err = get_eval_error(w,xtest,ptest,c1test,order)
        error1.append(err)   
#         print order,w
    print 'error evaluation:',error1
    print 'k-fold CV:',error0
    plt.figure()
    plt.scatter(order_list,error0,marker='o',color='r',label='k-fold CV')
    plt.scatter(order_list,error1,marker='*',color='b',label='error evaluation')
    plt.plot(order_list,error0,'r');plt.plot(order_list,error1,'b')
    plt.xlabel(r'different order')
    plt.legend()
    plt.grid(True)
    plt.savefig('./data/raw_ls.png')
    plt.show()
#     os.system('pause')
    
if __name__ == '__main__':
    raw_ls_regression()

