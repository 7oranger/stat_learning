# -*- coding:utf-8 -*-
'''
Created on 2017年2月9日

@author: RenaiC
'''
import random
import numpy as np
def update_2D_dict(thedict, key_a, key_b, val): 
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}}) 
        

def split_data(data,labels,k_fold = 10):
    assert k_fold > 1,"K-fold: invalid k"
    kf=float(1.0/k_fold)
    num_sample = data.shape[0] #178
    num_test = int(num_sample*kf)
    #print num_test
    #print num_sample
    index = range(num_sample)
    random.shuffle(index)
    #print  len(index)
    #index_cp = copy.deepcopy(index)
    index_test = {}
    index_train = {}
    for i in range(k_fold): # 0:K_fold 
        if i == k_fold-1: # last part,if k-fold=10,remaining 25 
            #tmp = range(num_test*i,num_sample)
            index_test[i]=[None]*(num_sample-num_test*(k_fold-1))
            index_train[i] = [None]*(num_test*(k_fold-1)) 
            j,m = 0, 0
            for x in xrange(num_sample):
                #print "x",x,"tmpx",tmp[x]
                if x >= num_test*i: #and x < num_test*(i+1):
                    #print i,j
                    index_test[i][j]= index[x]
                    j = j+1
                else:
                    index_train[i][m]= index[x] 
                    m = m+1
            #index = copy.deepcopy(index_cp)
        else:
            #tmp = range(num_test*i,num_test*(i+1))
            index_test[i]=[None]*num_test
            index_train[i] = [None]*(num_sample-num_test) 
            #print "k-fold",i
            #print 'index',index
            j,m = 0, 0
            for x in xrange(num_sample):
                if x >= num_test*i and x < num_test*(i+1):
                    #print i,j
                    index_test[i][j]= index[x]
                    j = j+1
                else:
                    index_train[i][m]= index[x] 
                    m = m+1
    return index_test,index_train
def get_weighted_error_rate(y_pred,y_real):
    age_sect = [1,18,25,35,45,50,56]
    error_rate = []
    for i in range(len(y_pred)):
        index1 = age_sect.index(y_pred[i])
        index2 = age_sect.index(y_real[i])
        error_rate.append( abs(index1-index2) )
    average_error_weight = np.mean(error_rate)
    
    return average_error_weight