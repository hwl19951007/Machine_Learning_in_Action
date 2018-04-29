# coding=utf-8
import random
from numpy import *
from  os import listdir


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


def img2vector(filename):
    """将文本化图形数据转化为向量"""
    returnVect = zeros((1, 1024))    # 转化为单列向量
    with open(filename) as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect



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


class optStruct:
    """建立一个类用于储存所需的所有数据，包括输入数据集，特征值，松弛变量和容错率，核函数类型"""
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn    # 数据集矩阵
        self.labelMat = classLabels    # 特征值矩阵
        self.C = C    # 松弛变量
        self.tol = toler    # 容错率
        self.m = shape(dataMatIn)[0]    # 数据集矩阵行数，即数据量
        self.alphas = mat(zeros((self.m, 1)))    # 根据矩阵行数初始化alpha为0
        self.b = 0                               # 初始化b为0
        self.eCache = mat(zeros((self.m, 2)))    # 第一列作为有效位，第二列为实际的E的值
        self.K = mat(zeros((self.m, self.m)))    # 初始化核K
        for i in range(self.m):                  # 计算所有数据的核K
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    """计算下标为k的数据误差Ek"""
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.K[:, k])) + oS.b
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
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
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
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]
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
    oS = optStruct(mat(dataMatin), mat(classLabels).transpose(), C, toler, kTup)    # 初始化数据结构
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


def kernelTrans(X, A, kTup):
    """通过核函数将数据转换为高维空间数据"""
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == "lin":    # 线性核函数，只进行内积
        K = X * A.T
    elif kTup[0] == "rbf":    # 高斯核函数，根据高斯核函数公式进行计算
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1]**2))    # 计算高斯核K
    else:
        raise NameError("Houston We Have a Problem That Kernel is not recognized")
    return K


def testRbf(k1=1.3):
    """测试函数，K1为高斯核函数的到达率"""
    dataArr, labelArr = loadDataSet('testSetRBF.txt')    # 加载训练集
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))    # 根据训练集计算b和alphas
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]    # 获取支持向量
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ("rbf", k1))    # 计算各个点的核
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b    # 根据支持向量的点，计算超平面，返回预测结果
        if sign(predict) != sign(labelArr[i]):                          # 与特征值做对比并统计错误个数
            errorCount += 1
    print("the training error rate is %f" % float(errorCount / m))    # 打印错误率
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')    # 加载测试集
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ("rbf", k1))    # 计算各个点的核
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b    # 根据支持向量的点，计算超平面，返回预测结果
        if sign(predict) != sign(labelArr[i]):                          # 与特征值做对比并统计错误个数
            errorCount += 1
    print("the test error rate is %f" % float(errorCount / m))    # 打印错误率


def img2vector(filename):
    """将文本化图形数据转化为向量"""
    returnVect = zeros((1, 1024))    # 转化为单列向量
    with open(filename) as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    """加载图片"""
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector("%s/%s" % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup = ("rbf, 10")):
    """测试函数"""
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is %f" % float(errorCount / m))
    dataArr, labelArr = loadImages('trainingDigits')
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is %f" % float(errorCount / m))


if __name__ == "__main__":
    # testRbf()
    testDigits(('rbf', 10))