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
import matplotlib.pyplot as plt

attr=['Alcohol','Malic acid', 'Ash', 'Alcalinity of ash',
    'Magnesium','Total phenols',
    'Flavanoids','Nonflavanoid phenols','Proanthocyanins',
    'Color intensity','Hue','OD280/OD315 of diluted wines',
    'Proline' 
    ]  
_k = 30
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

def KNNClassification(x_new, x_train, y_train, k):  
    if k<1:
        return None
    ##each time, process one new sample and return the predicted label
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
        #accumulate from 0. Each time the label appears, its 
        label_count[vote_label] = label_count.get(vote_label, 0) + 1 

    maxCount = 0  
    predicted_label = None
    # find the max vote_label
    for key, value in label_count.items():  
        if value > maxCount:  
            maxCount = value  
            predicted_label = key  
    return predicted_label   

def wine_classification(_k):
    assert _k>1,"Invalid k!" 
    data,labels=load_data()
    # split data into to parts. To perform 5-fold cross validation, test_size is 20%
    x_train, x_test, y_train, y_test = train_test_split(
                                                         data, labels, test_size=0.2, random_state=0)
    num_test = x_test.shape[0]  
    result=[]  # to store the result of accuracy for different k
    assert _k>1,"Invalid k!"     
    for k in range(1,_k+1):
        matchCount = 0  
        misMatch=0
        for i in xrange(num_test):  
            predict = KNNClassification(x_test[i], x_train, y_train, k)  
            
            if predict == y_test[i]:  
                matchCount += 1  
            else:
                misMatch+=1
                #print "Predict   "  +str(predict)+ "    Real   "  + str(y_test[i])
        accuracy = float(matchCount) / num_test  
        result.append(accuracy)
        #print "accuravy of", k, "is", accuracy*100,"%"
        #print 'The classify accuracy is: %.2f%%' % (accuracy * 100)  
        #print "MisMatch  "+str(misMatch)+"  numbers"+"\n"
    result_percent=[x*100 for x in result]
    print "accuracy of knn-classification for various k in terms of the wine data"
    for i in xrange(len(result)):
        print "k=",(i+1), "%.2f%%" %(result_percent[i])
    plt.figure()
    plt.plot(range(1,_k+1),result,'-o')
    plt.savefig('plot_of_k.png')
    plt.show()
    

if __name__ == "__main__":
    wine_classification(_k) 