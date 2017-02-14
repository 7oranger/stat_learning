# -*- coding:utf-8 -*-
'''
Created on 2016年10月28日
stat leaning Ex3-2
@author: RENAIC225
'''
import random,math,os
import matplotlib.pyplot as plt
import numpy as np

# -*- coding:utf-8 -*-

import random,math,os
import matplotlib.pyplot as plt
import numpy as np

def cal_sin(x):
    y = []
    for i in xrange(len(x)):
        y .append( math.sin(2*math.pi*x[i]))
    return y
def get_random():
    x = random.uniform(-1,1)
    err = random.gauss(0, 1)*0.1 # mean = 0 , var = 1
#     print err
    y = math.sin(2*math.pi*x) + err
    return x,y
def get_phi(x,order):
    height = len(x)
    a = np.zeros([height,order])
    for i in xrange(height):
        for j in xrange(order):
            a[i,j]=math.pow(x[i],j)
    return a
def generate_and_fit():
    num_data_set = 100
    num_points = 25
    data_set_y = []
    data_set_x = []
    points_x = []
    points_y = []
    order = 7+1
#     lamda = 0.01
    lamda_l = xrange(10)
    lamda_list= [math.pow(10,-1*i) for i in lamda_l]
   
    for i in xrange(num_data_set):
        'prepare dataset'
#         f = open(r'.\data\sin_test.txt',"w+")
        for j in xrange(num_points):
            x,y=get_random()
            points_x.append(x)
            points_y.append(y)
#             f.write(str(x)+'\t'+str(y)+'\n')
#         f.close()
#         os.system('pause')
        data_set_y.append(points_y)
        data_set_x.append(points_x)
        points_x = [] 
        points_y = [] 
    
    fig = plt.figure(figsize=(6,12))
    cnt =  0  
    r_square = []
    for my_lambda  in lamda_list:
        cnt = cnt + 1
        ax = fig.add_subplot(5,2,cnt)
        x = np.arange(-1, 1, 0.02) 
        y_sin=cal_sin(x)
        for i in xrange(num_data_set):
            'perform ridge regression'
            phi = get_phi(data_set_x[i],order)
            tmp0 = np.linalg.inv(np.dot(phi.T,phi)+my_lambda*np.eye(order))
            tmp1 = np.dot(tmp0,phi.T)
            omega = np.dot(tmp1, np.array(data_set_y[i]) )
            y_p = np.dot(get_phi(x,order),omega)
            ax.plot(x,y_p,'r')
            if my_lambda == 0.0001: # calculate Correlation coefficient
                r_square.append( np.corrcoef(y_p,y_sin,rowvar=0)[0,1] )
            #plt.scatter( (data_set_x[i]) , (data_set_y[i]), marker='.' )
        
        ax.plot(x,y_sin,'b')
#         ax.plot(x,y_sin,'b',label=r'$sin(2\pi x)$')
#         plt.legend('fitted')
#         ax.legend(loc='upper center')
#         plt.xlabel('x')
#         plt.ylabel('y')
        t = 'lambda='+str(my_lambda )+ ' for ' + r'$sin(2\pi x)$'
        plt.title(t)
        plt.savefig('lambda='+str(my_lambda ) +'.png')
#         plt.show()
    print np.mean(np.array(r_square)) # average Correlation coefficient
    plt.savefig('error' +'-0.001-.png')  
    plt.show()

if __name__ == '__main__':
    generate_and_fit()

