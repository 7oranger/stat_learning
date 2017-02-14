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
    return xtrain,ctrain,xtest,ptest,c1test # 200 training sample  6821 for testing 
def get_phi(x,order):
    'order =  1 2 3 ...,x has two element x[0] and x[1]'
    height = len(x)
    a = np.zeros([height,order*2+1])
    for i in xrange(height):
        for j in xrange(order+1):# 0 ~ order
            a[i,j]=math.pow(x[i,0],j)
    for i in xrange(height):
        for j in range(1,order+1):# 1 ~ order
            a[i,j]=math.pow(x[i,1],j)
    return a
def get_w(phi,order,y,my_lambda):
    tmp0 = np.linalg.inv(np.dot(phi.T,phi)+my_lambda*np.eye(2*order+1))
    tmp1 = np.dot(tmp0,phi.T)
    w = np.dot(tmp1, y )
    return w
    #y_p = np.dot(get_phi(x,order),omega)

def get_w_and_error(x,y,order,my_lambda):
    'give x and y, find the error by K fold-cv. error = average(each turn)'
    K = 5
    a = len(x)
    test_num = 1./K * a# K-fold CV num of validation set
    #train_num = a -  test_num # K-fold CV num of training set
    index = range(a)
    random.shuffle(index)  # 打乱
    x = x[index]
    y = y[index]
    error_kcv = []
    w = []
    for i in xrange(5):
        start = test_num*i # start point and end point of testing set
        end = test_num*(i+1)
        x_te = x[start:end] #test(validation)
        y_te = y[start:end] #train
        x_tr = np.vstack( (x[0:start],x[end:]) ) #the rest
        y_tr = np.concatenate( (y[0:start],y[end:]) ) #vstack
        xtr_phi = get_phi(x_tr,order)
        w0 =  get_w(xtr_phi,order,y_tr,my_lambda)
        w.append(w0) # store all w
        ### for validation
        y_predict = np.dot(get_phi(x_te,order),w0)#np.transpose(w0)
        error_kcv.append(compare(y_predict,y_te))
    wy = get_w(get_phi(x,order),order,y,my_lambda)
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
        if np.sign( predicted_y[i] ) == -1: # calssified as FALSE
            error = error + ptest[i]*c1test[i]
        else:
            error = error + ptest[i]*(1-c1test[i])
    return error 

def ridge_ls_regression():
    xtrain,ctrain,xtest,ptest,c1test = load_data()
    print ctrain.shape
    for i in xrange(len(ctrain)):
        if ctrain[i] == 0:
            ctrain[i] = -1
    order_list = range(1,10) # 1 ~ 9
    xxx = range(-6,7)
    lamda_list= [math.pow(10,1*i) for i in xxx]
    error0 = []
    error1 = []
    for my_lambda in lamda_list:
        order = 6
        w, err = get_w_and_error( xtrain,ctrain,order,my_lambda)
        error0.append(err)   
        err = get_eval_error(w,xtest,ptest,c1test,order)
        error1.append(err)   
#         print order,w
    print 'error evaluation:',error1
    print 'k-fold CV:',error0
    plt.figure()
    plt.scatter(xxx,error0,marker='o',color='r',label='k-fold CV')
    plt.scatter(xxx,error1,marker='*',color='b',label='error evaluation')
    plt.plot(xxx,error0,'r');plt.plot(xxx,error1,'b')
    plt.xlabel(r'$log_{10}(\lambda )$')
    plt.legend()
    plt.grid(True)
    plt.savefig('./data/ridge_ls.png')
    plt.show()
#     os.system('pause')
    
if __name__ == '__main__':
    ridge_ls_regression()

