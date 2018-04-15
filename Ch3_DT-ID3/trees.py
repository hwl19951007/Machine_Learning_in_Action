#coding=utf-8
from math import log
import operator
import treePlotter
import pickle


def calcShannonEnt(dataSet):
    """计算香农熵，度量数据集无序程度"""
    numEntries = len(dataSet)
    labelCounts = {}
    for fectVec in dataSet:    # 计算各类特征值出现的总数
        currentLabel = fectVec[-1]    # 获取数据特征值
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:    # 计算各特征值香农熵并通过for-loop求和
        prob = float(labelCounts[key])/numEntries    # 该类特征值出现概率
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def creatDataSet():
    """创建简单的鱼鉴定数据集"""
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """按照给定特征划分数据集  参数分别为未划分的数据集，划分数据集的特征，需要返回的特征的值
       将具有该特征值的数据选中，并剔除该特征值，返回都具有该特征值的更精简的数据集"""
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:    # 选中有该特征值的数据集
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])    # 切片
            retDataSet.append(reducedFeatVec)    # 存储至返回list
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """选择最好的数据集划分方式"""
    numFeatures = len(dataSet[0]) - 1    # 特征值数量
    baseEntorpy = calcShannonEnt(dataSet)    # 原数据集熵值
    baseInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]    # 获取该特征值的所有取值    使用列表推导来创建新的列表
        uniqueVals = set(featList)    # 删除重复，统计该特征值下的所有取值
        newEntorpy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntorpy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntorpy - newEntorpy
        if infoGain > baseInfoGain:
            baseInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """当决策树叶子内类别不唯一，选择最多的类别作为分类"""
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return classCount


def createTree(dataSet, labels):
    """递归算法创建决策树"""
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):    # 列表内第一个类别标签的数量等于列表长度，则类别完全相同
        return classList[0]
    if len(dataSet[0]) == 1:    # 使用完了所有特征，则返回最多的类别标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)    # 最优特征
    bestFeatLabel = labels[bestFeat]    # 最优特征的类别标签
    myTree = {bestFeatLabel: {}}    # 使用类别标签创建决策树
    subLabels=labels[:]    # 复制当前特征标签列表，防止修改原始列表内容
    del(subLabels[bestFeat])     # 删除被使用的类别标签
    featValues = [example[bestFeat] for example in dataSet]    # 获取特征值，并删除重复
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """使用决策树的分类函数"""
    firstSides = list(inputTree.keys())  # 需将字典转化为list后才能提取键值
    firstStr = firstSides[0]  # 找到第一个元素，为当前分类标签
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)    # 将字符串标签转为索引
    for key in secondDict.keys():  # 通过递归获取叶节点的数目
        if testVec[featIndex] == key:
            if isinstance(secondDict[key], dict):
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    """储存决策树"""
    with open(filename, 'wb') as f:
        pickle.dump(inputTree, f)


def grabTree(filename):
    """获取决策树文件"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def lenses():
    """创建隐形眼镜类型预测的决策树并绘图"""
    with open('lenses.txt') as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    treePlotter.createPlot(lensesTree)
    print(lensesTree)
    return lensesTree


myData, labels = creatDataSet()
# print(myData, '\n', labels)
# print(calcShannonEnt(myData))
# print(splitDataSet(myData, 0, 0))
# print(chooseBestFeatureToSplit(myData))
# print(createTree(myData, labels))
myTree = treePlotter.retrieveTree(0)
print(classify(myTree, labels, [1, 1]))
storeTree(myTree, 'classifierStorage.txt')
print(grabTree('classifierStorage.txt'))
lenses()