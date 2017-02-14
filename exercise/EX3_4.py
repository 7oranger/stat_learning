# -*- coding:utf-8 -*-
from __future__ import  division
'''
Created on 2016-12-04
Naive Bayesian with laplace smoothing
@author: RenaiC
'''
import numpy as np
import operator
x1= np.array([1, 1, 1, 1, 1, 2,2 ,2 ,2 ,2, 3])
x2= np.array(['s','m','m','s','s','s','m','m','l','l','l'])
Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1 ,1])
L = len(Y)
data = []
for i in xrange(L):
    data.append([x1[i],x2[i],Y[i]])
def get_dic(xx):
    xxd = {}
    for item in xx:
        if xxd.has_key(item):
            xxd[item] += 1
        else:
            xxd[item] = 1
    return xxd
def update_2D_dict(thedict, key_a, key_b, val): 
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}}) 
def get_frequency(x,y,dim):
    'dim=1: x1, dim=2:x2'
    cnt =  0
    for i in xrange(L):
        if data[i][dim-1] == x and data[i][2] == y:
            cnt = cnt + 1 
    return cnt
def naive_bayes(x_new,laplace_alpha):
    'input a test sample, if you donnot want to use laplace smothing,set alpha = 0'
    'naive bayes: 2 attributes'
    'Requirement： lables of all attributes cannot be the same, i.e. the label of all x must be distinguished'
    global Y, x1, x2, L, data
    yd = get_dic(Y)
    y_class = np.unique(Y)
    x1_class = np.unique(x1)
    x2_class = np.unique(x2) 
#     x_class = np.hstack((x1_class,x2_class))
    y_p = {} #计算Y的先验概率
    for key,value in yd.items():
        y_p[key] = value/ L
    
    dic_2D = {}  # P(x|y)
    for x in x1_class:
        for y in y_class:
            t = get_frequency(x, y, 1)
            val = (t+laplace_alpha)/(yd[y]+len(y_class)*laplace_alpha)
            update_2D_dict(dic_2D, x, y, val)
    for x in x2_class:
        for y in y_class:
            t = get_frequency(x, y, 2)   
            val = (t+laplace_alpha)/(yd[y]+len(y_class)*laplace_alpha)
            update_2D_dict(dic_2D, x, y, val)    
    # test the naive classification
    x1_new = x_new[0]
    x2_new = x_new[1]
    pred = {}
    for y in y_class:
        pred[y] = dic_2D[x1_new][y] * dic_2D[x2_new][y] * y_p[y]
    sorted_pred= sorted(pred.items(), key=operator.itemgetter(1))
    print sorted_pred
    return sorted_pred[1][0]
    
if __name__ == "__main__":
    x = [1,'m']
    y = naive_bayes(x,1)
    print 'Result of sample x:',x,' belongs to kind y=' ,y
    y = naive_bayes(x,0)
    print 'Result of sample x:',x,' belongs to kind y=' ,y