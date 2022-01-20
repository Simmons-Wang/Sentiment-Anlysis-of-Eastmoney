import numpy as np
import os

os.chdir("C:\\Users\\Simmons\\PycharmProjects\\数据挖掘大作业")


def sigmoid(z):
    """
    :param z: 输入
    :return: 激活值
    """
    return 1/(1 + np.exp(-z))


def relu(z):
    """
    :param z: 输入
    :return: 激活值
    """
    return np.array(z>0)*z


def sigmoidBackward(dA, cacheA):
    """
    :param dA: 同层激活值
    :param cacheA: 同层线性输出
    :return: 梯度
    """
    s = sigmoid(cacheA)
    diff = s*(1 - s)
    dZ = dA * diff
    return dZ


def reluBackward(dA, cacheA):
    """
    :param dA: 同层激活值
    :param cacheA: 同层线性输出
    :return: 梯度
    """
    Z = cacheA
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ


def iniPara(laydims):
    """
    :param laydims: 输入的结构（字典）
    :return:  随机初始化的参数字典
    """
    np.random.seed(1)
    parameters = {}
    for i in range(1, len(laydims)):
        parameters['W'+str(i)] = np.random.randn(laydims[i], laydims[i-1])/ np.sqrt(laydims[i-1])
        parameters['b'+str(i)] = np.zeros((laydims[i], 1))
    return parameters


def forwardLinear(W, b, A_prev):
    Z = np.dot(W, A_prev) + b
    cache = (W, A_prev, b)
    return Z, cache


def forwardLinearActivation(W, b, A_prev, activation):
    Z, cacheL = forwardLinear(W, b, A_prev)
    cacheA = Z
    if activation == 'sigmoid':
        A = sigmoid(Z)
    if activation == 'relu':
        A = relu(Z)
    cache = (cacheL, cacheA)
    return A, cache


def forwardModel(X, parameters):
    layerdim = len(parameters)//2
    caches = []
    A_prev = X
    for i in range(1, layerdim):
        A_prev, cache = forwardLinearActivation(parameters['W'+str(i)], parameters['b'+str(i)], A_prev, 'relu')
        caches.append(cache)
        
    AL, cache = forwardLinear(parameters['W'+str(layerdim)], parameters['b'+str(layerdim)], A_prev)
    caches.append(cache)
    
    return AL, caches


def computeCost(AL, Y):
    """
    :param AL: 输出层的激活输出
    :param Y: 实际值
    :return: 代价函数值
    """
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    return cost


def linearBackward(dZ, cache):
    W, A_prev, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def linearActivationBackward(dA, cache, activation):
    cacheL, cacheA = cache
    if activation == 'relu':
        dZ = reluBackward(dA, cacheA)
        dA_prev, dW, db = linearBackward(dZ, cacheL)
    elif activation == 'sigmoid':
        dZ = sigmoidBackward(dA, cacheA)
        dA_prev, dW, db = linearBackward(dZ, cacheL)
    return dA_prev, dW, db


def backwardModel(AL, Y, caches):
    layerdim = len(caches)
    Y = Y.reshape(AL.shape)
    L = layerdim
    
    diffs = {}
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    currentCache = caches[L-1]
    dA_prev, dW, db =  linearBackward(dAL, currentCache)
    diffs['dA' + str(L)], diffs['dW'+str(L)], diffs['db'+str(L)] = dA_prev, dW, db
    
    for l in reversed(range(L-1)):
        currentCache = caches[l]
        dA_prev, dW, db =  linearActivationBackward(dA_prev, currentCache, 'relu')
        diffs['dA' + str(l+1)], diffs['dW'+str(l+1)], diffs['db'+str(l+1)] = dA_prev, dW, db
        
    return diffs


def updateParameters(parameters, diffs, learningRate):
    layerdim = len(parameters)//2
    for i in range(1, layerdim+1):
        parameters['W'+str(i)] -= learningRate*diffs['dW'+str(i)]
        parameters['b'+str(i)] -= learningRate*diffs['db'+str(i)]
    return parameters


def finalModel(X, Y, layerdims, learningRate=0.01, numIters=5000,pringCost=False):
    np.random.seed(1)
    costs = []
    parameters = iniPara(layerdims)
    
    for i in range(0, numIters):
        AL, caches = forwardModel(X, parameters)
        cost = computeCost(AL, Y)
        
        diffs = backwardModel(AL,Y, caches)
        parameters = updateParameters(parameters,diffs, learningRate)
    
        if pringCost and i%100 == 0:
            costs.append(np.sum(cost))
    return parameters


def predict(X, parameters):
    a, b = forwardModel(X, parameters)
    return a, b
