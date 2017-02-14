
# -*- coding:utf-8 -*-
'''
Created on 2016年10月28日
stat leaning Ex3-3
percerption original form
@author: RENAIC225
'''
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:49:04 2016

@author: fengxinhe
"""
import copy
from matplotlib import pyplot as pl
from matplotlib import animation as ani

w=[0,0,0]  #weight vector w[2] = b
# b=0      #bias 
yita=0.1   #learning rate
# data=[[(1,4),1],[(0.5,2),1],[(2,2.3), 1], [(1, 0.5), -1], [(2, 1), -1],[(4,1),-1],[(3.5,4),1],[(3,2.2),-1]]
# data=[[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
data = [ [(3,3),1],  [(4,3),1],  [(1,1),0]   ]
record=[]

"""
if y(wx+b)<=0,return false; else, return true
"""

    
"""
update the paramaters w&b
"""
def predict(vec):
    y=w[0]*vec[0][0]+w[1]*vec[0][1]-w[2]
    if y>0.5:
        y_p = 1
    else:
        y_p = 0
    return y_p
def update(vec,y):
    global w,b,record
    y=w[0]*vec[0][0]+w[1]*vec[0][1]-w[2]
    w[0] = w[0]+yita*(vec[1]-y)*vec[0][0]
    w[1] = w[1]+yita*(vec[1]-y)*vec[0][1]
    w[2] = w[2]+yita*(vec[1]-y) #doesn't work
#     w[0]=w[0]+yita*vec[1]*vec[0][0]
#     w[1]=w[1]+yita*vec[1]*vec[0][1]
#     w[2]=w[2]+yita*vec[1]
    record.append(w)
#     print record
#     w[0]=w[0]+yita*vec[1]*vec[0][0]
#     w[1]=w[1]+yita*vec[1]*vec[0][1]
#     b=b+yita*vec[1]
#     record.append([copy.copy(w),b])

"""
check and calculate the whole data one time
if all of the input data can be classfied correctly at one time, 
we get the answer
""" 
def perceptron():
    count = 0
#     print data
    for item in data:
        y = predict(item)
        if y == item[1]:
            count = count +1
        else:
            count = 1
            update(item,y)
    if count >= len (data):
        print 'count',count,'len',len (data)
        return 1
    else:
        return 0
#     for ele in data:
#         print ele
#         flag=sign(ele)
#         if not flag>0:
#             count=1
#             update(ele)
#         else:
#             count+=1
#     if count>=len(data):
#         return 1
#         
      
        
if __name__ == "__main__":
    i = 1
    while 1:
        print 'i:',i
        print 'w',w
        a = perceptron()
        i = i+1
        if a > 0:
            break
    print 'record', record
# x1=[]
# y1=[]
# x2=[]
# y2=[]
# 
# #display the animation of the line change in searching process
# fig = pl.figure()
# ax = pl.axes(xlim=(-1, 5), ylim=(-1, 5))
# line,=ax.plot([],[],'g',lw=2)
# 
# def init():
#     line.set_data([],[])
#     for p in data:
#         if p[1]>0:
#             x1.append(p[0][0])
#             y1.append(p[0][1])
#         else:
#             x2.append(p[0][0])
#             y2.append(p[0][1])
#     pl.plot(x1,y1,'or')
#     pl.plot(x2,y2,'ob')
#     return line,
# 
# 
# def animate(i):
#     global record,ax,line
#     w=record[i][0]
#     b=record[i][1]
#     x1=-5
#     y1=-(b+w[0]*x1)/w[1]
#     x2=6
#     y2=-(b+w[0]*x2)/w[1]
#     line.set_data([x1,x2],[y1,y2])
#     return line,
#     
# animat=ani.FuncAnimation(fig,animate,init_func=init,frames=len(record),interval=1000,repeat=True,
#                                    blit=True)
# pl.show()
# # animat.save('perceptron.gif', fps=2,writer='imagemagick')  
#         
    