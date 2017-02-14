# -*- coding:utf-8 -*-

'''
Created on 2016-10-18
Project of statistical learning : project1 -- classification for wine data
reference here:
http://scikit-learn.org/stable/modules/cross_validation.html
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
@author: RENAIC225
'''

#from sklearn import preprocessing
from sklearn.cross_validation import train_test_split  
import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt


attr=['Alcohol','Malic acid', 'Ash', 'Alcalinity of ash',
    'Magnesium','Total phenols',
    'Flavanoids','Nonflavanoid phenols','Proanthocyanins',
    'Color intensity','Hue','OD280/OD315 of diluted wines',
    'Proline' 
    ]  
_knn = 30
_k_fold = 5
_new_dim = 6
def load_data():
    wine_data = np.loadtxt('wine.data', delimiter=",") # local directory
    #a,b = wine_data.shape
    #print a,b # 178 lines, 14 columns .  178 examples; 13 attributes, the first column is its class label
    labels = np.array(wine_data[:,0])  # the first column reprents label
    dataset= np.array(wine_data[:,1:]) # the rest columns are data
    #print labels.shape,labels.size  #178
    #print dataset.shape,dataset.size  #  178*13
    #print labels[50:100]
    #print dataset[0:13,0:13]
    return dataset,labels

def pca(raw_data, new_dim): # pca
    if new_dim >= raw_data.shape[1]:
        return raw_data,raw_data

    # cal covvariance and eigen vector and eigen value
    cov=np.cov(raw_data.transpose())
    eigval, eigvec=np.linalg.eig(cov)
    
    eigval_index = np.argsort(eigval) # ascending order
    eigval_index = eigval_index[::-1] # reverses
    eigval_index = eigval_index[:new_dim] # get the first new_index 
    eigvec_used = eigvec[:,eigval_index]
    low_dim_data=np.dot(raw_data,eigvec_used) # cal new low-dimentional data
    recovered_data = np.dot(low_dim_data, eigvec_used.T)    #re-construct the data
    return low_dim_data,recovered_data
 
def split_data(data,labels,k_fold): # split the dataset into training set and testing set for k-fold cv
    assert k_fold > 1,"K-fold: invalid k"
    kf=float(1.0/k_fold)
    num_sample = data.shape[0] #178
    num_test = int(num_sample*kf)
    index = range(num_sample)
    random.shuffle(index)
    
    index_test = {} #dict to store index of test data
    index_train = {} #dict to store index of train data
    for i in range(k_fold): # 0:K_fold 
        j,m = 0, 0
        if i == k_fold-1: # last part,if k-fold=10,remaining 25 
            index_test[i]=[None]*(num_sample-num_test*(k_fold-1))
            index_train[i] = [None]*(num_test*(k_fold-1)) 
         
            for x in xrange(num_sample): # split  into 2 parts
                if x >= num_test*i: #and x < num_test*(i+1):
                    index_test[i][j]= index[x]
                    j = j+1
                else:
                    index_train[i][m]= index[x] 
                    m = m+1
        else:  # the first k-fold-1 datasets
            index_test[i]=[None]*num_test
            index_train[i] = [None]*(num_sample-num_test) 
            for x in xrange(num_sample):
                if x >= num_test*i and x < num_test*(i+1):
                    index_test[i][j]= index[x]
                    j = j+1
                else:
                    index_train[i][m]= index[x] 
                    m = m+1
                    
    # save the index of test to verify 
    fw=open("index_test.txt","w+")
    for item in index_test:
        fw.write("fold id:")
        fw.write("%s\n" % item)
        for v in xrange(len(index_test[item])):
            fw.write("%s\n" % index_test[item][v])
    fw=open("index_train.txt","w+")
    # save the index of train to verify 
    for item in index_train:
        fw.write("fold id:")
        fw.write("%s\n" % item)
        for v in xrange(len(index_train[item])):
            fw.write("%s\n" % index_train[item][v])
    #for i in xrange(k_fold):pickle.dump(index_test[i], fw)
    
    return index_test,index_train
 
    
def KNNClassification(x_new, x_train, y_train, k):  # each time, process one new sample and return the predicted label
    if k<1:
        return None
    num_train = x_train.shape[0] # shape[0] stands for the num of row  
    # tile(A, Size): Construct an array by repeating A for Size times  
    diff = np.tile(x_new, (num_train, 1)) - x_train # get diff, same dimension
    squared_diff = diff ** 2 # squared 
    squared_dist = np.sum(squared_diff, axis = 1) # sum, by row  
    distance = squared_dist ** 0.5   # root, distance between the new sample with other training set 
    dist_index = np.argsort(distance)  # sort the distance, in ascending order
  
    label_count = {} # define a dictionary as an accumulator
    for i in xrange(k):  
        vote_label = y_train[dist_index[i]]  # y label: 1 2 3
        # accumulate from 0. Each time the label appears, increase by 1 
        label_count[vote_label] = label_count.get(vote_label, 0) + 1 

    maxCount = 0  
    predicted_label = None
    # find the max vote_label
    for key, value in label_count.items():  
        if value > maxCount:  
            maxCount = value  
            predicted_label = key  
            
    return predicted_label   

def wine_classification(_knn, k_fold):
    assert _knn > 0,"Invalid k: knn!"   
    assert k_fold > 1,"Invalid k: k-fold !" 
    
    src_data,labels=load_data()
    np.savetxt("raw_data.txt",src_data)
    data,recovered_data = pca(src_data,_new_dim)
    #print src_data.shape,data.shape,recovered_data.shape
    # save all these data to verify
    np.savetxt("pca_data_recovered.txt",recovered_data)
    np.savetxt("pca_data.txt",data)
    np.savetxt('diff.txt',recovered_data-src_data)
    # split data into to parts. To perform 5-fold cross validation, test_size is 20%
    #x_train, x_test, y_train, y_test = train_test_split(
    #                                                     data, labels, test_size=0.2, random_state=0)
    index_test,index_train = split_data(data,labels,k_fold)
    
    result_k= {}
    result_final=[]
    for k in  range(1,_knn+1): # different knn-k
        result_k[k-1]=[] 
        for i in xrange(_k_fold): # different fold
            x_train = data[index_train[i],:]
            y_train = labels[index_train[i]]
            x_test = data[index_test[i]]
            y_test = labels[index_test[i]]
            num_test = x_test.shape[0]  
            # to store the result of accuracy for different k, begin with 0
            #assert _k>1,"Invalid k!"     
            matchCount = 0  
            misMatch=0
            for n in xrange(num_test):  
                predict = KNNClassification(x_test[n], x_train, y_train, k)   
                if predict == y_test[n]:  # bingo
                    matchCount += 1  
                else: # dismatch
                    misMatch+=1
            accuracy = float(matchCount) / num_test * 100. # change to percentile
            print " k-fold:",i+1,"of",k_fold, accuracy,"%"
            result_k[k-1].append(accuracy) # accuracy at one fold

        result_final.append(np.mean(result_k[k-1]))
        
        print "Accuracy of knn-classification for various k in terms of the wine data"
        print 'Accuracy with KNN-K=',k,': ',result_final[k-1],"%"
        print "--------------------------------------------------------"
        
    max_val=max(result_final) # get the highest accuracy
    max_index = result_final.index(max_val)
    np.savetxt("accuracy_of_different_k_1",result_final)
    
    plt.figure()
    plt.plot(range(1,_knn+1),result_final,'-o')
    plt.savefig('plot_of_k_cv_pca.png')
    plt.show()
    return max_val,max_index+1

if __name__ == "__main__":
    print "start of knn classification of wine data\n"
    print "-------------------------------------------------------\n"
    accuracy,best_k,=wine_classification(_knn, _k_fold) 
    print "accuracy:%.2f%%"%accuracy,"\nbest_k:",best_k