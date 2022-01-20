import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers


def pcan(dataX, datasTad, n):
    tr_cov = np.cov(datasTad, rowvar=0)
    eigenValue, eigenVector = np.linalg.eig(tr_cov)  # 求得特征值，特征向量
    sorceEigenValue = np.argsort(eigenValue)  # 特征值下标从小到大的排列顺序
    nPcaEigenVector = sorceEigenValue[-n:]  # 最大的n个特征值的下标
    pcaEigenVector = eigenVector[nPcaEigenVector]  # 选取特征值对应的特征向量
    pcaEigenValue = eigenValue[nPcaEigenVector]
    sum_value = np.sum(eigenValue)
    pcaEigenValue = pcaEigenValue/sum_value
    PCAX = np.dot(dataX, pcaEigenVector.T)  # 得到降维后的数据
    return PCAX, pcaEigenVector, pcaEigenValue


def build_model(i, lr,x_train):
    model = models.Sequential()  # 这里使用Sequential模型
    model.add(layers.Dense(6, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(i, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(1))
    # 编译网络
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
    return model


def r_square(y, p):
    Residual = sum((y - p) ** 2)  # 残差平方和
    total = sum((y - np.mean(y)) ** 2)  # 总体平方和
    R_square = 1 - Residual / total  # 相关性系数R^2
    return R_square

