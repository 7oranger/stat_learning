# -*- coding: utf-8 -*-
'''
Created on 2017年2月6日

@author: RenaiC
'''
from pandas.core import indexing
'''
task:
Task 1: User profiling. You need to predict the gender and age of users
based on their ratings on movies
'''
import numpy as np
import re,pickle,time,json,random, timeit
from sklearn.cross_validation import train_test_split  
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from Project3_util import *
import matplotlib.pyplot as plt
######################### read raw data
start = time.clock()
start_time = timeit.default_timer()
print 'loading raw data and transfer into dict...'
movies = np.loadtxt('./data/movies.dat', delimiter = "::" , dtype = 'string' );# two column dtype='float' usecols=(0,1) ,
# print movies.shape # 读三列报错啊 # 电影编号 电影名字 有乱码 删除后即可 3776行 #3883种电影
# print movies[1] #scipy.io.loadmat : reads MATLAB data files
# movie_info = []
# for i in range(movies.shape[0]):
#     movie_info.append(  [movies[i][0]]+movies[i,2].split('|')  ) # ['1', 'Animation', "Children's", 'Comedy']
movie_dict = {}
for i in range(movies.shape[0]):
    movie_dict[ movies[i][0] ] = movies[i,2].split('|')   # {'1': ['Animation', "Children's", 'Comedy']}
# print len(movie_info)# 3883
# print movie_info[0]
genre_type = movies[:,2]
genre_unique = np.unique(genre_type)
genre_dict = {} # type：frequency
for i in genre_unique:
# obtain genre information
    #ttt =  re.split(r'|',i)
    ttt = i.split('|')
    for word in ttt:
        if genre_dict.has_key(word):
            genre_dict[word] += 1
        else:
            genre_dict[word] = 1    # 从0改为1  
genre_list = sorted(genre_dict.items(), key=lambda e:e[1], reverse=True)
genre = [x[0] for x in genre_list ] #共 18 种题材
# print genre,len(genre) 
# print len(movies) 
ratings = np.loadtxt('./data/ratings.dat', delimiter = "::" ,usecols=(0,1,2), dtype = 'int' );
rating_dict = {}
for i in range(ratings.shape[0]):
    update_2D_dict(rating_dict,ratings[i][0],ratings[i][1],ratings[i][2]) #userID movieID rating

uname = ['./data/users.dat'+str(i) for i in range(10)]
user_info = []
for filename in uname:
    t = np.loadtxt(filename, delimiter = "::" ,usecols=(0,1,2,3) , dtype = 'string')
#     print t.shape #(604L, 4L)
    user_info.append(t)

user = np.concatenate( [user_info[i] for i in range(len(user_info)) ], axis=0) #(6040L, 4L)
all_user = user[:,0]
user_dict = {}
for i in  range(len(user)):
    user_dict[user[i][0]] = user[i][1:]
# print user_dict['3'] #['M' '25' '15']
# print len(user_info) # 10
# print user_info[1].shape
# print user_info[0][0] #['9' 'M' '25' '17']

# uinfo = np.ones( (10,604,4), dtype='int' ) #转换为int 类型
# for i in  range(len(user_info)):
#     for j in  range(user_info[i].shape[0]):
#         for k in [0,2,3]:
#             uinfo[i][j][k] = int(user_info[i][j][k])
#         if user_info[i][j][1] == 'M':
#             uinfo[i][j][1] = 1
#         else:
#             uinfo[i][j][1] = 0

# for          
# with open('./data/user_info.pkl','w') as f:
#     pickle.dump(uinfo,f)
# with open('./data/movie_info.pkl','w') as f:
#     pickle.dump(movies,f)
# with open('./data/rating_info.pkl','w') as f:
#     pickle.dump(ratings,f)
# print len(uinfo)#  10
# print uinfo[1].shape
# print uinfo[0,0]# ['9' 'M' '25' '17']  #    [ 9  1 25 17]                   

###################prepare input data for classifier
# rating_dict = {}
print 'prepare input data...'
X = []
Y1 = [] #sex
Y2 = [] #age
# for kU in user_dict.keys():#遍历所有用户。所有用户ID：string
for i in range(len(all_user)):
    kU = all_user[i]
    userId = int(kU)
    #['M' '25' '15']
    user_age =  int(user_dict[kU][1])
    user_occupation = int(user_dict[kU][2])
    user_sex = 0
    if user_dict[kU][0] == 'M':
        user_sex = 1
    for kM in rating_dict[userId].keys(): #遍历所有电影。rating都是int类型的 
        movieIdInt = kM
        movieId = str(kM) # 电影id的string类型
        movie_rate = rating_dict[userId][kM]
        movie_type = movie_dict[movieId]# a list of strings
        movieT = [0]*len(genre) # 18维
        for index,item in enumerate(genre):
        #将电影类型转换成一个18维的bool  list
            if item in movie_type:
                movieT[index] = 1
                 
        movieRate = rating_dict[userId][kM] #评分
        x = [userId,user_occupation,movieIdInt,movie_rate]
        x.extend(movieT)
        X.append(x)
        Y1.append(user_sex)
        Y2.append(user_age)
    

index = range(len(Y1))
random.shuffle(index)

# r = np.random.permutation(len(Y1)) # 随机地从全排列中选取一个，实现 shuffle
X = np.array(X)[index, :] 
Y1 = np.array(Y1)[index] 
Y2 = np.array(Y2)[index] 
def classifyY1(data,labels,_k_fold = 10):
    index_test,index_train = split_data(data,labels,_k_fold)
    acc=[]
    process_time = []
    for i in xrange(_k_fold): # different fold
        #print k,i
        '''
        print len(index_train[i])
        print len(index_test[i])
        print index_train[i]
        print index_test[i]
        '''
        time1 = timeit.default_timer()
        x_train = data[index_train[i]]
        y_train = labels[index_train[i]]
        x_test = data[index_test[i]]
        y_test = labels[index_test[i]]
        classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.001)
        classifier.fit(x_train,y_train)
        score = classifier.score( x_test, y_test)
        time2 = timeit.default_timer()
        acc.append(score)
        process_time.append((time2-time1))
    return process_time,acc

def classifyY2(data,labels,_k_fold = 10):
    index_test,index_train = split_data(data,labels,_k_fold)
    weighted_error_rate=[]
    acc=[]
    process_time = []
    for i in xrange(_k_fold): # different fold
        #print k,i
        '''
        print len(index_train[i])
        print len(index_test[i])
        print index_train[i]
        print index_test[i]
        '''
        print 'round:',i
        time1 = timeit.default_timer()
        x_train = data[index_train[i]]
        y_train = labels[index_train[i]]
        x_test = data[index_test[i]]
        y_test = labels[index_test[i]]
        classifier = RandomForestClassifier()
        classifier.fit(x_train,y_train)
        score = classifier.score( x_test, y_test)
        y_pred = classifier.predict(x_test)
        weighted_error_rate.append (get_weighted_error_rate(y_pred,y_test) )
        acc.append(score)
        time2 = timeit.default_timer()
        process_time.append((time2-time1))
    return process_time,acc,weighted_error_rate

process_time1,acc1 = classifyY1(X, Y1, 10)
print 'process_time1',np.mean(process_time1)
print 'acc1',np.mean(acc1)

plt.figure()
plt.plot(range(1,10+1),process_time1,'-o')
plt.savefig('gender-time-consuming.png')
plt.show()

plt.figure()
plt.plot(range(1,10+1),acc1,'-o')
plt.savefig('gender-accuracy.png')
plt.show()

process_time2,acc2,weighted_error_rate2 = classifyY2(X, Y2, 10)
print 'process_time2',np.mean(process_time2)
print 'acc2',np.mean(acc2)
print 'weighted_error_rate2',np.mean(weighted_error_rate2)

plt.figure()
plt.plot(range(1,10+1),process_time2,'-o')
plt.savefig('age-time-consuming.png')
plt.show()

plt.figure()
plt.plot(range(1,10+1),acc2,'-o')
plt.savefig('age-accuracy.png')
plt.show()

plt.figure()
plt.plot(range(1,10+1),weighted_error_rate2,'-o')
plt.savefig('age-weighted-error-rate.png')
plt.show()
####################  test for single fold
#--------------sex-------------------
        
        
# with open('./data/X-.txt','w+') as f:
#     json.dump(X.tolist(),f,indent=4, sort_keys=False, separators=(',','.'))
# with open('./data/y1-.txt','w+') as f:
#     json.dump(Y1.tolist(),f,indent=4, sort_keys=False, separators=(',','.'))
# with open('./data/y2-.txt','w+') as f:
#     json.dump(Y2.tolist(),f,indent=4, sort_keys=False, separators=(',','.'))
# x1_train, x1_test, y1_train, y1_test = train_test_split( X, Y1, test_size=0.1, random_state=1)     
# x2_train, x2_test, y2_train, y2_test = train_test_split( X, Y2, test_size=0.1, random_state=1)     
# ######################### fit 
# print 'fitting by logistic regression classifier...'
# classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, penalty='l2', random_state=None, tol=0.001)
# 
# #--------------sex-------------------
# print '\nresult about Y1: gender'
# print 'number of training sample:',len(x1_train)
# print 'number of testing sample:',len(x1_test)
# classifier.fit(x1_train,y1_train)
# print 'score:',classifier.score( x1_test, y1_test)
# #------------------age-----------------------
# print '\nresult about Y2: age'
# print 'number of training sample:',len(x2_train)
# print 'number of testing sample:',len(x2_test)
# classifier.fit(x2_train,y2_train)
# y2_pred = classifier.predict(x2_test)
# print y2_pred[0:50]
# print y2_train[0:50]
# print 'score:',classifier.score( x2_test, y2_test)
# # c = [(y2_pred[i],y2_test[i]) for i in range( len(y2_test) )]
# 
# # with open('./data/compare_age.json','w+') as f:
# #     json.dump(c,f)
# 
# print 'random forest'
# # cla = OneVsRestClassifier(SVC(kernel='linear'))
# cla = RandomForestClassifier()
# cla.fit(x2_train,y2_train)
# r = cla.predict(x2_test)
# print r[0:50]
# print 'score:',cla.score( x2_test, y2_test)
# # print 'svm classifier'
# # clf = SVC()  # class   
# # clf.fit(x2_train,y2_train)  # training the svc model  
# #   
# # result = clf.predict(x2_test) # predict the target of testing samples  
# # print result[0:50]
# end = time.clock()
# end_time = timeit.default_timer()
# print 'time elapsed(s):',end-start
# print 'processing time(s):',(end_time-start_time)

print 'End of the programme'