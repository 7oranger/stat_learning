# -*- coding: utf-8 -*-
from __future__ import division
from matplotlib import pyplot as plt

w=[0,0,0]  #weight vector w[2] = b
yita=0.1   #learning rate
data = [ [(3,3),1],  [(4,3),1],  [(1,1),-1]  ]
result=[]  #trace for w

def predict(x):
    y=w[0]*x[0][0]+w[1]*x[0][1]+w[2]
    if y>0:
        y_p = 1
    else:
        y_p = -1
    return y_p

def update(x):
    global w,b,record
    w[0]=w[0]+yita*x[1]*x[0][0]
    w[1]=w[1]+yita*x[1]*x[0][1]
    w[2]=w[2]+yita*x[1]
    result.append(w)
    
def perceptron():
    'one turn'
    count = 1
    for item in data:
        y = predict(item)
        if y == item[1]:
            'right'
            count = count +1
        else:
            'wrong,then update'
            count = 1
            update(item)
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
    print 'Final w:',result[-1]
    #draw a figure to demostrate
    plt.figure()
    plt.title('perceptron  demo')
    for item in data:
        if item[1]==1:
            plt.scatter(item[0][0],item[0][1],marker = 'x', color = 'y')
        else:
            plt.scatter(item[0][0],item[0][1],marker = 'o', color = 'b')
    
    x = range(-3,6)
    if w[1] == 0:
        t = -1*w[2]/w[0]
        x = [t]*6
        y = xrange(6)
    else:
        y = [ (-1*w[2]-w[0]*t)/w[1] for t in x]
    plt.plot(x,y,'r')
    plt.savefig('perception-origin.png')  
    plt.show()
#     print 'record\n', result

