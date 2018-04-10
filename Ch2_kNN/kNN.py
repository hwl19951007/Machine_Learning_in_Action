#coding=utf-8
from numpy import *
import operator    # 运算符模块
import matplotlib    # 可视化工具
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    '''创建数据集'''
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'B', 'C', 'D']
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''k-近邻算法'''
    dataSetSize = dataSet.shape[0]     # shape读取矩阵行列长度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet    # tile用于将输入点inX，复制出等同dataSet的份数并做矩阵减法
    sqDiffMat = diffMat**2    # 求各维度距离的平方
    sqDistances = sqDiffMat.sum(axis=1)    # axis为0为普通相加，为1为行向量相加
    distances = sqDistances**0.5    # 开根号求出样本点与其他点的距离
    sortedDistIndices = distances.argsort()    # argsort返回数值从小到大的索引值排序
    classCount = {}
    for i in range(k):    # 选择距离最小的前k个点
        voteIlable = labels[sortedDistIndices[i]]    # 根据排序后的下标返回距离最小的的前k个标签
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1    # 计算各个标签出现的频率
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)    # 对字典进行排序
    return sortedClassCount[0][0]    # 返回排序后是一个元祖，返回第一个元祖的第一个值。


def file2matrix(filename):
    '''将文本形式的数据转化为矩阵形式'''
    with open(filename) as f:    # 打开文本
        arrayOLines = f.readlines()    # 读取所有数据行
    numberOlines = len(arrayOLines)    # 获取数据行数（数据量m）
    returnMat = zeros((numberOlines, 3))    # 创建返回矩阵，numberOfLines行，3列
    classLabelVector = []    # 返回的分类标签向量
    index = 0    # 行的索引值
    for line in arrayOLines:
        line = line.strip()    # 删除空白符
        listFromLine = line.split('\t')    # 根据"/t"作为分隔符切片
        returnMat[index, :] = listFromLine[0:3]    # 将数据前三列提取至返回矩阵内，即特征矩阵
        classLabelVector.append(int(listFromLine[-1]))    # 获取最后一列即喜欢程度信息并存储
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    '''归一化特征值'''
    minVals = dataSet.min(0)    # 存放每列最小值，参数0使得可以从列中选取最小值，而不是当前行
    maxVals = dataSet.max(0)    # 存放每列最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))    # 初始化归一化矩阵大小
    m = dataSet.shape[0]    #获取数据总数
    normDataSet = dataSet - tile(minVals, (m, 1))    #原始值减去最小值
    normDataSet = normDataSet/tile(ranges, (m, 1))    #除以范围获得归一化值
    return normDataSet, ranges, minVals


def datingClassTest():
    '''测试预测结果正确率'''
    hoRatio = 0.1    #设定测试集数量为总量的0.1
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    norMat, ranges, minVals = autoNorm(datingDataMat)    #获取数据并换算为归一化特征值、范围、最小值
    m = norMat.shape[0]    #获取数据量
    numTestVecs = int(m * hoRatio)        #整型测试集的数量
    errorCount = 0.0    #定义错误结果计数器
    for i in range(numTestVecs):
        classifierResult = classify0(norMat[i, :], norMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)    #分类
        print("the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, datingLabels[i]))    #返回结果
        if(classifierResult != datingLabels[i]): errorCount += 1    #如果错误，则计数器加一
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))    #返回错误率


def classifyPerson():
    '''约会网站喜欢结果预测'''
    resultList = ["not at all", "in small does", "in large does"]    #返回结果集
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))    #获取三变量
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    norMat, ranges, minVals = autoNorm(datingDataMat)    #获取数据并归一化
    inArr = array([ffMiles, percentTats, iceCream])    #整合输入数据
    classifierResult = classify0((inArr-minVals)/ranges, norMat, datingLabels, 3)    #预测
    print("You will probably like this person: ", resultList[classifierResult-1])

# group, labels = createDataSet()
# print(classify0([0, 0], group, labels, 3))
# datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
# print(datingDataMat, datingLabels)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
# plt.show()
# norMat, ranges, minVals = autoNorm(datingDataMat)
# print(norMat, ranges, minVals)
#datingClassTest()
#classifyPerson():


def img2vector(filename):
    '''将文本化图形数据转化为向量'''
    returnVect = zeros((1, 1024))    #转化为单列向量
    with open(filename) as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    '''数字识别算法'''
    hwLabels = []    #定义数字特征
    trainingFileList = listdir("trainingDigits")    #训练数据集文件列表
    m = len(trainingFileList)    #训练数据量
    trainingMat = zeros((m, 1024))    #训练集矩阵定义
    for i in range(m):
        fileNameStr = trainingFileList[i]    #获取训练集样本文件名
        fileStr = fileNameStr.split('.')[0]    #去除文件尾.txt
        classNumStr = int(fileStr.split('_')[0])    #去除文件名后部排序号
        hwLabels.append(classNumStr)    #获取训练集特征值
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)    #获取训练集矩阵并储存
    testFileList = listdir('testDigits')    #测试集数据集文件列表
    errorCount = 0.0    #定义错误结果计数器
    mTest = len(testFileList)    #测试集数据数量
    for i in range(mTest):
        fileNameStr = testFileList[i]    #获取测试集样本文件名
        fileStr = fileNameStr.split('.')[0]    #去除文件尾.txt
        classNumStr = int(fileStr.split('_')[0])    #去除文件名后部排序号
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)    #获取测试集单个样本矩阵用于测试
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)    #识别手写数字
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if(classifierResult != classNumStr): errorCount += 1    #若错误计数器加一
    print("\nthe total number of errors is: %d" % errorCount)
    print('\nthe total error rate is:%f' % (errorCount/float(mTest)))

returnVect = img2vector('testDigits/0_13.txt')
print(returnVect[0, 0:31])
handwritingClassTest()

