# -*- coding:utf-8 -*-
'''
Created on 2016年12月6日
ref 
Machine Learning in Action Ch9
http://blog.csdn.net/namelessml/article/details/52595066
http://blog.csdn.net/lipengcn/article/details/50260033
@author: RenaiC
'''
from numpy import *  
import matplotlib.pyplot as plt
import random as rd
import os
from numpy.linalg.linalg import norm
my_lambda = 0.0001 #正则项
my_order = 3 + 1
def loadDataSet(fileName) :  #测试用
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines() :
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):  #二分,按特征的值来二分，此处只有一个特征x，则 feature一直为0
#     print feature,value
#     os.system('pause')
#     排过序后其实可以直接切分
    mat0 = dataSet[ nonzero(dataSet[:,feature] > value)[0],:]  
    mat1 = dataSet[ nonzero(dataSet[:,feature] <= value)[0],:]  
    return mat0,mat1  

def createTree(dataSet,  ops) :   # 其模型是一个my_order阶的多项式
    # 将数据集分成两个部分，若满足停止条件，chooseBestSplit将返回None和某类模型的值
    # 若不满足停止条件，chooseBestSplit()将创建一个新的Python字典，并将数据集分成两份，
    # 在这两份数据集上将分别继续递归调用createTree()函数
    feat, val = chooseBestSplit(dataSet, ops)
    if feat == None : 
        return val
    retTree = {} # a dict: a tree
    retTree['spInd'] = feat # 该树的划分节点(或特征)
    retTree['spVal'] = val # 该树划分节点的值
    lSet, rSet = binSplitDataSet(dataSet, feat, val) #split into two part 
    retTree['left'] = createTree(lSet, ops)
    retTree['right'] = createTree(rSet, ops)
    return retTree

# 回归树的切分函数，构建回归树的核心函数。目的：找出数据的最佳二元切分方式。如果找不到
# 一个“好”的二元切分，该函数返回None并同时调用createTree()方法来产生叶节点，叶节点的值也将返回None。
# 如果找到一个“好”的切分方式(ls)，则返回特征编号(x，0)和切分特征值(y)。
def chooseBestSplit(dataSet, ops) :
    # tolS是容许的误差下降值, tolN是切分的最小样本数
    tolS = ops[0]; 
    tolN = ops[1]
    # 如果剩余特征值的数目为1，那么就不再切分而返回
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1 :
        return None, modelLeaf(dataSet)
    m,n = shape(dataSet)
    # 当前数据集的误差
    S = modelErr(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1) : # 遍历所有特征(即 x，对于一个特征的回归来说，feature index 永远是0)
        for splitVal in set( dataSet[:, featIndex].T.tolist()[0] ) :#在 x 中遍历所有的
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN) : continue
            newS = modelErr(mat0) + modelErr(mat1)
            if newS < bestS :
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果切分数据集后效果提升不够大，那么就不应该进行切分操作而直接创建叶节点
    if (S - bestS) < tolS :
        return None, modelLeaf(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 检查切分后的子集大小，如果某个子集的大小小于用户定义的参数tolN，那么也不应切分。
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN) :
        return None, modelLeaf(dataSet)
    # 如果前面的这些终止条件都不满足，那么就返回切分特征和特征值。
    return bestIndex, bestValue

# 给定x, 计算 phi 矩阵
def get_phi(x,order):
    height = len(x)
    a = zeros([height,order])
    for i in xrange(height):
        for j in xrange(order):
            a[i,j]=math.pow(x[i],j)
    return mat(a)
# 将数据格式化成目标变量Y和自变量X。X、Y用于岭回归求解系数ws。
def linearSolve(dataSet) :
    m,n = shape(dataSet) 
    X = mat(dataSet[:, 0:n-1])
    X_phi = mat(get_phi(X,my_order))
    Y = mat(dataSet[:, -1])
    tmp = X_phi.T*X_phi + my_lambda*mat( eye(my_order) ) # 添加正则项
    ws = tmp.I*(X_phi.T*Y)
    return ws, X, Y

# 叶节点存储 系数 ws
def modelLeaf(dataSet) :
    ws, X, Y = linearSolve(dataSet)
    return ws

# 在给定的数据集上计算误差。least-square
def modelErr(dataSet) :
    ws, X, Y = linearSolve(dataSet)
    yHat = get_phi(X, my_order) * ws
    return sum(power(Y-yHat, 2)) + my_lambda*linalg.linalg.norm(ws)# 真实值-预测值的平方

def modelTreeEval(model, inDat) :
    #  y =Phi*ws
    X = get_phi(inDat, my_order)
    return float(X*model)
 
def isTree(obj):  
    return (type(obj).__name__ == 'dict')  

# 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
# modeEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
# 此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
def treeForecast(tree, inData, modelEval=modelTreeEval) :
    if not isTree(tree) : 
        return modelTreeEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal'] :
        if isTree(tree['left']) :
            return treeForecast(tree['left'], inData)
        else : 
            return modelTreeEval(tree['left'], inData)
    else :
        if isTree(tree['right']) :
            return treeForecast(tree['right'], inData)
        else :
            return modelTreeEval(tree['right'], inData)

# 多次调用treeForeCast()函数，以向量形式返回预测值，在整个测试集进行预测
def createForecast(tree, testData,) :
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m) :
        yHat[i,0] = treeForecast(tree, mat(testData[i]))
    return yHat

def cal_sin(x):
    y = []
    for i in xrange(len(x)):
        y .append( math.sin(2*math.pi*x[i]))
    return y
def get_random():
    x = rd.uniform(-1,1)
    err = rd.gauss(0, 1)*0.1 # mean = 0 , var = 1
    y = math.sin(2*math.pi*x) + err
    return x,y
def generate_data():
    num_data_set = 100
    num_points = 25
    data_set = []
    points = []
    for i in xrange(num_data_set):
        for j in xrange(num_points):
            x,y=get_random()
            points.append([x,y])
        points=sorted(points, key=lambda points : points[0]) 
        data_set.append(points)
        points = [] 
    return data_set
if __name__ == '__main__':  
    #     trainMat = mat(loadDataSet(r'.\data\sin_train.txt'))
#     testMat = mat(loadDataSet(r'.\data\sin_test.txt'))
    data_set = generate_data()
#     plt.figure()
    r_square = []
    xt =mat(arange(-1, 1, 0.02)) 
  
   # 在 100 个数据集上做实验,用第x组构建树，用x+1组测试，测试指标用 预测值和真实值的相关系数
    for i in xrange(100):
        trainMat = mat(data_set[i])
#         testMat = mat(data_set[i+1])
#         print "create model tree..."
        myModTree = createTree(trainMat,ops=(2,4))
#         print "predicting new sample.."
        yHat = createForecast(myModTree,xt.T) 
#          yHat = createForecast(myModTree,testMat[:,0]) 
#         x=testMat[:,0]
        y_sin = cal_sin(arange(-1, 1, 0.02))
        r_square.append(corrcoef(yHat,y_sin,rowvar=0)[0,1])
#         plt.plot(x,yHat,'y')
#         plt.scatter(x,y,c='r',marker='o')
#         plt.scatter(x,yHat,c='r')
#         plt.plot(x,yHat,'r')
    print "correlation coefficient(average for 100 times) between real y and predicted y:",mean(array(r_square))
    x_sin = arange(-1, 1, 0.02) 
    y_sin = cal_sin(x_sin)
#     plt.plot(x_sin,y_sin,c='b')
#     plt.show()
  