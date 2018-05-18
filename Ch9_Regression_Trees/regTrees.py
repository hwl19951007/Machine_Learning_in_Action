from numpy import *


def loadDataSet(fileName):
    """读取数据并以浮点形式储存"""
    dataMat = []
    with open(fileName) as f:
        for line in f.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))    # 将每行映射成浮点数
            dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """根据特征切分数据集合成为两个子集"""
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """生成叶节点，返回目标变量的均值"""
    return mean(dataSet[:, -1])


def regErr(dataSet):
    """误差估计函数，计算目标变量的总方差"""
    return var(dataSet[:, -1]) * shape(dataSet)[0]    # 平均方差乘以总样本数


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """计算当前数据集最好的二分切分方式，回归树构建的核心"""
    tolS = ops[0]                                          # 容许的误差下降值
    tolN = ops[1]                                          # 切分的最少样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:        # 如果当前的所有值相等，则直接建立叶节点（set函数会将所有重复值删除）
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)                                                    # 默认最后一个特征为最佳切分特征，计算误差估计
    bestS = inf                                                             # 预定义最佳误差
    bestIndex = 0                                                           # 预定义最佳切分特征的索引值
    bestValue = 0                                                           # 预定义最佳切分特征值
    for featIndex in range(n - 1):                                          # 遍历所有特征
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):         # 遍历所有特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)      # 根据特征和特征值进行切分
            if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:              # 如果切分后数据集小于tolN，则退出本次循环
                continue
            newS = errType(mat0) + errType(mat1)                            # 否则计算当前误差
            if newS < bestS:                                                # 如果当前误差值更小，更新最佳误差、特征、特征值
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:                                                  # 如果误差下降，则直接建立叶节点
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)             # 获取最佳切分后的数据集
    if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:                      # 如果切分后数据集依然小于tolN，则直接建立叶节点
        return None, leafType(dataSet)
    return bestIndex, bestValue                                             # 如果符合切分条件，则返回最佳切分特征及特征值


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """构建树函数，是一个递归函数。输入为数据集，建立叶节点的函数，误差计算函数，其他参数"""
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    """判断输入变量是否是一棵树（是否为字典变量）"""
    return isinstance(obj, dict)


def getMean(tree):
    """对树进行塌陷处理（即返回树的平均值）"""
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])    # 递归直到找到叶节点，计算均值。返回整棵树的均值
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """后剪枝"""
    if shape(testData)[0] == 0:    # 判断当前测试集是否为空，为空则对当前树做塌陷处理（即返回树的平均值）
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):    # 如果当前树有左子树或右子树，则以数据集最优切分方法切分当前测试集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):    # 对当前左子树进行剪枝处理
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):    # 对当前右子树进行剪枝处理
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['right']) and not isTree(tree['left']):    # 如果当前左右节点均为叶节点，则切分当前测试集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))    # 计算当前节点未合并时的误差
        treeMean = (tree['left'] + tree['right']) / 2.0    # 计算当前节点合并后的均值
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))    # 计算当前节点合并后的误差
        if errorMerge < errorNoMerge:    # 如果合并后的误差小于合并前的误差，则合并
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


# ##########################################模型树###################################################


def linearSolve(dataSet):
    """将数据集格式化成目标变量Y 和自变量X """
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('矩阵为奇异矩阵，不能求逆，请尝试增大ops的第二个值（切分的最少样本数）')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    """生成叶节点，回归系数矩阵ws"""
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """计算数据集的误差"""
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


# ##########################################树回归与标准回归的比较###################################################


def regTreeEval(model, inDat):
    """回归树的预测值"""
    return float(model)


def modelTreeEval(model, inDat):
    """模型树的预测值"""
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    """对于输入的单个数据点，treeForeCast返回一个预测值。modelEval：预测结果计算函数"""
    if not isTree(tree):                                             # 如果达到叶节点，计算预测值
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:                        # 样本值大于阈值，进入左子树进行预测，否则进入右子树
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    """对测试集进行预测"""
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    # myDat = loadDataSet('ex0.txt')
    # myMat = mat(myDat)
    # print(createTree(myMat))
    # myDat2 = loadDataSet('ex2.txt')
    # myMat2 = mat(myDat2)
    # myTree = createTree(myMat2, ops=(0, 1))
    # print(myTree)
    # myDatTest = loadDataSet('ex2test.txt')
    # myMat2Test = mat(myDatTest)
    # pruneTree = prune(myTree, myMat2Test)
    # print(pruneTree)
    # myMat2 = mat(loadDataSet('exp2.txt'))
    # myTree = createTree(myMat2, modelLeaf, modelErr, (1, 10))
    # print(myTree)
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops=(1, 20))    # 回归树
    yHat = createForeCast(myTree, testMat[:, 0])
    print('回归树的相关系数为：', corrcoef(yHat, testMat[:, -1], rowvar=0)[0, 1], '\n')
    myTree = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))    # 模型树
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print('模型树的相关系数为：', corrcoef(yHat, testMat[:, -1], rowvar=0)[0, 1], '\n')
    ws, X, Y = linearSolve(trainMat)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print('线性回归的相关系数为：', corrcoef(yHat, testMat[:, -1], rowvar=0)[0, 1], '\n')
