# coding=utf-8
import random
from numpy import *


def loadDataSet(fileName):
    """便利函数。 从文件中读取数据集"""
    dataMat = []
    labelMat = []
    with open(fileName) as f:
        for line in f.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])    # 获取X1，X2
            labelMat.append(float(lineArr[2]))    # 获取类别标签
    return dataMat, labelMat

def selectJrand(i, m):
    """随机选择alpha。  i是alpha的下标，m是alpha的总数，选出一个不等于i 的j 。"""
    j = i
    while j == i:    # 选择一个不等于i的值
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """根据边界修剪alpha的值，H 和L 分别是alpha在约束条件下的上下限值"""
    if aj > H:
        aj = H
    elif L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """简化版smo算法，C：松弛变量，toler：容错率"""
    dataMatrix = mat(dataMatIn)    # 转化为numpy矩阵
    labelMat = mat(classLabels).transpose()    # 转化为numpy矩阵
    b = 0    # 初始化参数b
    m, n = shape(dataMatrix)    # 获取数据量和特征量即dataMatrix的维度
    alphas = mat(zeros((m, 1)))    # 初始化alpha的参数
    iterNum = 0    # 初始化迭代次数
    while iterNum < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 计算误差Ei multiply为矩阵内对应元素相乘
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - classLabels[i]
            # 优化alpha，更设定一定的容错率。如果误差过大则调整alpha。如果alpha已经为C或0说明已被调整，不值得再次优化
            if (labelMat[i] * Ei < -toler and alphas[i] < C) or (labelMat[i] * Ei > toler and alphas[i] > 0):
                j = selectJrand(i, m)
                # 步骤1：计算误差Ej
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - classLabels[j]
                # 保存更新前的aplpha值，使用深拷贝
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 计算上下界
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue    # 跳出本次循环
                # 步骤3：计算eta
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eat >= 0")
                    continue
                # 步骤4：更新alpha_j
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                # 步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 步骤6：更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 步骤7：更新b_1和b_2
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 步骤8：根据b_1和b_2更新b
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                # 更新优化次数，打印统计信息
                alphaPairsChanged += 1
                print("iterNum: %d i : %d, pairs changed %d" % (iterNum, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iterNum += 1
        else:
            iterNum = 0
        print("iteration number : %d" % iterNum)
    return b, alphas


class optStruct:
    """建立一个类用于储存所需的所有数据，包括输入数据集，特征值，松弛变量和容错率"""
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn    # 数据集矩阵
        self.labelMat = classLabels    # 特征值矩阵
        self.C = C    # 松弛变量
        self.tol = toler    # 容错率
        self.m = shape(dataMatIn)[0]    # 数据集矩阵行数，即数据量
        self.alphas = mat(zeros((self.m, 1)))    # 根据矩阵行数初始化alpha为0
        self.b = 0                               # 初始化b为0
        self.eCache = mat(zeros((self.m, 2)))    # 第一列作为有效位，第二列为实际的E的值


def calcEk(oS, k):
    """计算下标为k的数据误差Ek"""
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    """内循环启发方式2，选择J，返回J、Ej"""
    maxK = -1    # 初始化
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]    # 更新Ei缓存
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]    # 返回E（误差）不为0的索引值 /// .A表示将矩阵转化为numpy数组
    if len(validEcacheList) > 1:    # 如果有不为0的E（误差）
        for k in validEcacheList:    # 遍历，找到最大的Ek
            if k == i:               # 不计算已被选中的i
                continue
            Ek = calcEk(oS, k)       # 计算Ek
            deltaE = abs(Ei - Ek)    # 计算detlaE = |Ei -Ek |
            if deltaE > maxDeltaE:    # 将 最大 E(误差)的索引值和delta获取
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)    # 如果没有不为0的E(误差)，随机选择J
        Ej = calcEk(oS, j)          # 计算Ej
    return j, Ej


def updateEk(oS, k):
    """计算Ek，并更新Ek缓存"""
    Ek = calcEk(oS, k)
    oS.eCache[k] = Ek


def innerL(i, oS):
    """完整Platt SMO算法"""
    # 步骤1：计算误差Ei
    Ei = calcEk(oS, i)    # 优化alpha,设定一定的容错率。
    if (oS.labelMat[i] * Ei < -oS.tol and oS.labelMat[i] * Ei < oS.C) or \
            (oS.labelMat[i] * Ei > oS.tol and oS.labelMat[i] * Ei > 0):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 步骤2：计算上下界L和H
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta >= 0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 步骤8：根据b_1和b_2更新b
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatin, classLabels, C, toler, maxIter, kTup=("lin", 0)):
    """"""
    oS = optStruct(mat(dataMatin), mat(classLabels).transpose(), C, toler)    # 初始化数据结构
    iterNum = 0    # 初始化当前迭代次数
    entireSet = True
    alphaPairsChanged = 0
    # 如果超过最大迭代次数或遍历数据集alphas无更新，退出循环
    while iterNum < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:    # 遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)    # 使用优化的SMO算法，如果有alphas被优化，则alphaPairsChanged加一
                print("fullSet, iterNum: %d i : %d, pairs changed %d" % (iterNum, i, alphaPairsChanged))
            iterNum += 1
        else:    # 遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]    # 遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)    # 使用优化的SMO算法，如果有alphas被优化，则alphaPairsChanged加一
                print("non-bound, iterNum: %d i : %d, pairs changed %d" % (iterNum, i, alphaPairsChanged))
            iterNum += 1
        if entireSet:    # 遍历一次后改为非边界遍历
            entireSet = False
        elif alphaPairsChanged == 0:     # 如果alpha没有更新,计算全样本遍历
            entireSet = True
        print("iteration number: %d" % iterNum)
    return oS.b, oS.alphas    # 返回SMO算法计算的b和alphas


def calcWs(alphas, dataArr, classLabels):
    """计算W"""
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w
















if __name__ == "__main__" :
    dataArr, labelArr = loadDataSet('testSet.txt')
    print(labelArr)
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # print(b, alphas[alphas > 0])
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    ws = calcWs(alphas, dataArr, labelArr)
    print(b, "\n", alphas[alphas > 0], "\n", ws)
    # a = 0
    # for i in range(100):
    #     if mat(dataArr)[2] * mat(ws) * labelArr[2] > 0:
    #         a += 1
    # print(a)





