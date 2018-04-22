# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    """便利函数。 从文件中读取数据集"""
    dataMat = []
    labelMat = []
    with open('testSet.txt') as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])    # 添加X0为1，获取X1，X2
            labelMat.append(int(lineArr[2]))    # 获取类别标签
    return dataMat, labelMat


def sigmoid(inX):
    """sigmoid函数"""
    return 1.0/(1 + exp(-inX))


def graAscent(dataMatIn, classLabels):
    """梯度上升算法"""
    dataMatrix = mat(dataMatIn)    # 转化为numpy的mat（矩阵）格式
    labelMat = mat(classLabels).transpose()    # 转化为numpy的mat（矩阵）格式，并转化为列向量
    m, n = shape(dataMatrix)    # 获取数据集的数量和特征数量（行数和列数）
    alpha = 0.001    # 学习速率
    maxCycles = 500    # 最大迭代次数
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)    # 梯度上升矢量化公式
        error = labelMat - h    # 偏离量
        weights = weights + alpha * dataMatrix.transpose() * error    # 调整最佳参数，继续迭代
    return weights


def plotBesatFit(wei):
    """绘制分类后的数据集和最佳拟合直线"""
    weights = array(wei)   # 将权重矩阵转化为数组
    dataMat, labelMat = loadDataSet()    # 获取数据矩阵和类别标签
    dataArr = array(dataMat)    # 将数据矩阵转化为numpy的数组
    n = shape(dataArr)[0]    # 获取数据个数
    xcord1 = []    # 数据点坐标集
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):    # 将数据坐标根据类别标签分别写入数据坐标集
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')    # 分别绘制两种特征值的数据点
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)    # 最佳拟合直线的横纵坐标
    y = (-weights[0] - weights[1] * x)/weights[2]    # 这里的y坐标实际上是X2的值，根据公式0=W0*X0+W1*X1+W2*X2计算，其中x0=1
    ax.plot(x, y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    """随机梯度上升算法"""
    m, n = shape(dataMatrix)    # 获取数据集的个数和特征数
    alpha = 0.01    # 学习速率
    weights = ones(n)    # 初始化回归系数
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))    # 计算当前样本的假设函数值
        error = classLabels[i] - h    # 计算误差
        weights = weights + alpha * error * dataMatrix[i]    # 更新回归系数
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """改进的随机梯度算法"""
    m, n = shape(dataMatrix)    # 获取数据集的个数和特征数
    weights = ones(n)    # 初始化回归系数
    for j in range(numIter):    # 控制迭代次数
        dataIndex = list(range(m))# 一个包含0到m-1的的list
        for i in range(m):    # 对所有样本做计算
            alpha = 4 / (1.0 + j + i) + 0.01    # 调整学习速率，随着迭代次数增加而减小对回归系数的影响，每次减小1 / (j + i)
            randIndex = int(random.uniform(0, len(dataIndex)))    # 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))    # 计算当前样本的假设函数值
            error = classLabels[randIndex] - h    # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]    # 更新回归系数
            del(dataIndex[randIndex])    # 删除使用过的样本
    return weights


def classifyVector(inX, weights):
    """根据计算的Sigmoid值来给输入数据集分类"""
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:    # 如果sigmoid的值大于0.5，则属于1，否则为0
        return 1.0
    else:
        return 0.0


def colicTest():
    """打开训练集和测试集，训练后用测试集测试，返回错误率"""
    trainingSet = []
    trainingLabels = []
    with open('horseColicTraining.txt') as frTrain:    # 获取训练数据集
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')    # split分割后的返回list
            lineArr = []
            for i in range(len(currLine) - 1):    # 将除最后一个类别标签外的特征数据加入到lineArr
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[-1]))    # 将类别标签添加到trainingLabels
    trainingWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)    # 用随机上升算法进行训练
    errorCount = 0
    numTestVec = 0.0
    with open('horseColicTest.txt') as frTest:    # 获取测试数据集
        for line in frTest.readlines():
            numTestVec += 1    # 统计测试集数量
            currLine = line.strip().split('\t')    # split分割后的返回list
            lineArr = []
            for i in range(len(currLine) - 1):    # 将除最后一个类别标签外的特征数据加入到lineArr
                lineArr.append(float(currLine[i]))
            if int(classifyVector(array(lineArr), trainingWeights)) != int(currLine[-1]):    # 判断分类结果是否正确
                errorCount += 1    # 若错误则错误计数器加一
    errorCount = float(errorCount) / numTestVec    # 计算错误率
    print("the error rate of this test is: %f" % errorCount)
    return errorCount


def multiTest():
    """多次训练并测试，返回更可信的平均错误率"""
    numTests = 10    # 训练十次
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is %f " % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # weights = graAscent(dataArr, labelMat)
    # print(weights)
    # plotBesatFit(weights)
    # weights = stocGradAscent0(array(dataArr), labelMat)
    # print(weights)
    # plotBesatFit(weights)
    # weights = stocGradAscent1(array(dataArr), labelMat)
    # print(weights)
    # plotBesatFit(weights)
    multiTest()