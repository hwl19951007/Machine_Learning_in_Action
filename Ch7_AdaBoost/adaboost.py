# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt


def loadSimpData():
    """加载简单的数据供测试"""
    dataMat = matrix([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """通过阈值比较，对数据进行分类，dimen(dimension)：特征种类，threshVal：阈值，threshIneq：不等号"""
    retArray = ones((shape(dataMatrix)[0], 1))    # 初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1    # 此时不等式为小于，如果小于阈值，则赋值为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1    # 此时不等式为大于，如果大于阈值，则赋值为-1
    return retArray


def buildStump(dataArr, classLabels, D):
    """遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树。D：样本权重"""
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf    # 初始化最小误差为正无穷大
    for i in range(n):    # 遍历所有特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps    # 通过特征值中的最大最小值来设置计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:               # 遍历大于和小于的情况'lt':less than 'gt':greater than
                threshVal = (rangeMin + float(j) * stepSize)    # 计算阈值
                predictdVals = stumpClassify(dataMatrix, i, threshVal, inequal)    # 计算分类结果
                errArr = mat(ones((m, 1)))                                         # 初始化误差矩阵
                errArr[predictdVals == labelMat] = 0                               # 分类结果正确则赋值为1
                weightedError = D.T * errArr                                       # 计算误差
                # print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f"
                #       % (i, threshVal, inequal, weightedError))
                if weightedError < minError:    # 找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictdVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineqal'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIter = 40):
    """基于单层决策树的AdaBoost训练过程"""
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)    # 初始化权重
    aggClassEst = mat(zeros((m, 1)))    # 每个数据点的类别估计预测值
    for i in range(numIter):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)    # 构建单层决策树
        # print("D:", D.T)
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))    # 计算弱分类器的权重alpha，使error不为0避免除零溢出
        bestStump['alpha'] = alpha                                   # 存储该弱分类器的权重alpha
        weakClassArr.append(bestStump)                               # 存储该弱分类器
        # print("classEst: ", classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)    # 计算e的指数项，这三行计算D
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst                                # 计算AdaBoost累计预测值
        # print("aggClassEst ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))    # 计算误差，sign()获取正负性
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:                                          # 如果总错误率为1，退出循环，返回
            break
    return weakClassArr, aggClassEst


def adaClassify(dataToClass, classifierArr):
    """AdaBoost分类函数"""
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineqal'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return sign(aggClassEst)


def loadDataSet(fileName):
    """便利函数。 从文件中读取数据集"""
    dataMat = []
    labelMat = []
    with open(fileName) as f:    # 获取训练数据集
        numFeat = len(f.readline().split('\t'))    # 获取特征和类别的和，其中特征数量为总和减一
    with open(fileName) as f:    # 获取训练数据集
        for line in f.readlines():
            lineArr = []
            curLine = line.strip().split('\t')    # split分割后的返回list
            for i in range(numFeat - 1):
                lineArr.append(float(curLine[i]))    # 将除最后一个类别标签外的特征数据加入到lineArr
            dataMat.append(lineArr)    # 获取数据样例
            labelMat.append(float(curLine[-1]))    # 获取类别标签
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    """绘制ROC曲线，predStrengths为分类器的预测强度"""
    cur = (1.0, 1.0)    # 绘制光标的位置
    ySum = 0.0    # 用于计算AUC
    numPosClas = sum(array(classLabels) == 1.0)    # 统计正类的数量
    yStep = 1 / float(numPosClas)    # 根据正类数量获取y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)    # 根据负类数量获取x轴步长
    sortedIndicies = predStrengths.argsort()    # 预测强度排序
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]    # 高度累加
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')    # 绘制ROC
        cur = (cur[0] - delX, cur[1] - delY)    # 更新绘制光标的位置
    ax.plot([0, 1], [0, 1], 'b--')    # 绘制随机猜测时得到的虚线
    plt.title('AdaBoost马疝病检测系统的ROC曲线', size=14)
    plt.xlabel('假阳率(False Positive Rate)')
    plt.ylabel('真阳率(True Positive Rate)')
    ax.axis([0, 1, 0, 1])
    print('AUC(Area Under the Curve)为：', ySum * xStep)
    plt.show()


if __name__ == "__main__":
    # dataMat, classLabels = loadSimpData()
    # D = mat(ones((5, 1))/5)
    # print(bulidStump(dataMat, classLabels, D))
    # classifierArr = adaBoostTrainDS(dataMat, classLabels, 30)
    # adaClassify([[0, 0], [5, 5]], classifierArr)
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
    print(aggClassEst)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    predictions = adaClassify(dataArr, classifierArray)
    errArr = mat(ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != mat(labelArr).T].sum() / len(dataArr) * 100))
    predictions = adaClassify(testArr, classifierArray)
    errArr = mat(ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != mat(testLabelArr).T].sum() / len(testArr) * 100))
    plotROC(aggClassEst.T, labelArr)