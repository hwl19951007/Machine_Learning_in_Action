# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


def loadDataSet(fileName):
    """便利函数。 从文件中读取数据集"""
    dataMat = []
    labelMat = []
    with open(fileName) as f:    # 获取训练数据集
        numFeat = len(f.readline().split('\t')) - 1    # 获取特征和类别的和，其中特征数量为总和减一
    with open(fileName) as f:    # 获取训练数据集
        for line in f.readlines():
            lineArr = []
            curLine = line.strip().split('\t')    # split分割后的返回list
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))    # 将除最后一个类别标签外的特征数据加入到lineArr
            dataMat.append(lineArr)    # 获取数据样例
            labelMat.append(float(curLine[-1]))    # 获取类别标签
    return dataMat, labelMat


def standRegres(xArr, yArr):
    """简单线性回归，计算回归系数"""
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:    # 若行列式为0说明该矩阵为奇异矩阵，不能求逆。
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I * (xMat.T * yMat)    # 计算最小的W并返回
    return ws


def plotRegression():
    """绘制回归曲线和数据点"""
    xArr, yArr = loadDataSet('ex0.txt')                                    # 加载数据集
    ws = standRegres(xArr, yArr)                                        # 计算回归系数
    xMat = mat(xArr)                                                    # 创建xMat矩阵
    yMat = mat(yArr)                                                    # 创建yMat矩阵
    # 计算相关系数
    yHat = xMat * ws
    print(corrcoef(yHat.T, yMat))
    xCopy = xMat.copy()                                                    # 深拷贝xMat矩阵
    xCopy.sort(0)                                                        # 排序
    yHat = xCopy * ws                                                     # 计算对应的y值
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            # 添加subplot
    ax.plot(xCopy[:, 1], yHat, c='red')                                # 绘制回归曲线
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=0.5)                # 绘制样本点
    plt.title('DataSet')                                                # 绘制title
    plt.xlabel('X')
    plt.show()


def lwlr(testPoint, xArr, yArr, k=1.0):
    """给定x空间中任意一点，计算对应的预测值yHat"""
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))    # 初始化权重对角矩阵
    for j in range(m):    # 遍历整个数据集，计算对于点每个样本的权重
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:    # 若行列式为0说明该矩阵为奇异矩阵，不能求逆。
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))    # 计算回归系数
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    """为数据集中的每个点调用lwls()函数，"""
    m = shape(testArr)[0]    # 获取测试数据集的大小
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)    # 对每个样本点进行预测
    return yHat


def plotlwlrRegression():
    """绘制多条局部加权回归曲线"""
    xArr, yArr = loadDataSet('ex0.txt')                                    # 加载数据集
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)                            # 根据局部加权线性回归计算yHat
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)                            # 根据局部加权线性回归计算yHat
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)                            # 根据局部加权线性回归计算yHat
    xMat = mat(xArr)                                                    # 创建xMat矩阵
    yMat = mat(yArr)                                                    # 创建yMat矩阵
    srtInd = xMat[:, 1].argsort(0)                                        # 排序，返回索引值
    xSort = xMat[srtInd][:, 0, :]
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')                        # 绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')                        # 绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')                        # 绘制回归曲线
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)             # 绘制样本点
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)             # 绘制样本点
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)             # 绘制样本点
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title('局部加权回归曲线,k=1.0')
    axs1_title_text = axs[1].set_title('局部加权回归曲线,k=0.01')
    axs2_title_text = axs[2].set_title('局部加权回归曲线,k=0.003')
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()


def rssError(yArr, yHatArr):
    """计算预测误差大小"""
    return ((yArr - yHatArr)**2).sum()


def plotabalonetest():
    """测试核K对学习结果的影响"""
    abX, abY = loadDataSet('abalone.txt')
    print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[0:99], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[0:99], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[0:99], yHat10.T))
    print('\t')
    print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[100:199], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[100:199], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[100:199], yHat10.T))
    print('\t')
    print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    ws = standRegres(abX[0:99], abY[0:99])
    yHat = mat(abX[100:199]) * ws
    print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))


def ridgeRegres(xMat, yMat, lam=0.2):
    """岭回归，计算回归系数"""
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:              # 不能保证lam不为0，因此依然要计算矩阵是否可逆
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    """岭回归测试"""
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 数据标准化
    yMean = mean(yMat, 0)            # 计算每列均值，因为y只有一列，因此求的是整体的均值
    yMat = yMat - yMean              # 减去均值进行标准化处理
    xMeans = mean(xMat, 0)           # 计算每列均值，即每个特征的均值
    xVar = var(xMat, 0)              # 计算每列方差，即每个特征的方差值
    xMat = (xMat - xMeans) / xVar    # 每个元素减去均值再除以方差，进行标准化处理
    numTestPts = 30                  # 30次不同的lambda测试
    wMat = zeros((numTestPts, shape(xMat)[1]))       # 初始化回归系数矩阵
    for i in range(numTestPts):                      # 改变lambda以计算回归系数
        ws = ridgeRegres(xMat, yMat, exp(i - 10))    # lambda以e的指数幂进行变化，以了解在lambda取值不同的情况下造成的影响
        wMat[i, :] = ws.T                            # 保存至一个矩阵内
    return wMat


def plotridgeRegres():
    """绘制岭回归系数矩阵"""
    abX, abY = loadDataSet('abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    print(redgeWeights, len(redgeWeights))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title('log(lambada)与回归系数的关系')
    ax_xlabel_text = ax.set_xlabel('log(lambada)')
    ax_ylabel_text = ax.set_ylabel('回归系数')
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


def regularize(xMat):
    """数据标准化"""
    inMat = xMat.copy()  # 数据深拷贝
    inMeans = mean(inMat, 0)    # 计算每列均值，即每个特征的均值
    inVar = var(inMat, 0)    # 计算每列方差，即每个特征的方差值
    inxMat = (inMat - inMeans) / inVar  # 每个元素减去均值再除以方差，进行标准化处理
    return inxMat


def stageWise(xArr, yArr, eps=0.01, numIter=100):
    """前向逐步线性回归算法（贪心算法，每一步都尽可能减少误差）eps为步长"""
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 数据标准化
    yMean = mean(yMat, 0)  # 计算每列均值，因为y只有一列，因此求的是整体的均值
    yMat = yMat - yMean  # 减去均值进行标准化处理
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIter, n))    # 初始化numIter次迭代的回归系数
    ws = zeros((n, 1))                 # 初始化回归系数矩阵
    wsTest = ws.copy()
    wsBest = ws.copy()
    for i in range(numIter):           # 迭代numIter次
        print(ws.T)
        lowestError = inf              # 初始化最小误差为无限大
        for j in range(n):                             # 遍历每个特征的回归系数
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign                # 微调回归系数
                yTest = xMat * wsTest                  # 计算微调后的预测值
                rssE = rssError(yMat.A, yTest.A)       # 计算微调后的误差值（平方误差）
                if rssE < lowestError:                 # 如果误差减小，则更新当前权重矩阵
                    lowestError = rssE
                    wsBest = wsTest
        ws = wsBest.copy()
        returnMat[i, :] = ws.T                         # 记录numIter次迭代的回归系数
    return returnMat


def plotstageWiseMat():
    """绘制前向逐步线性回归系数矩阵"""
    xArr, yArr = loadDataSet('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title('前向逐步回归:迭代次数与回归系数的关系')
    ax_xlabel_text = ax.set_xlabel('迭代次数')
    ax_ylabel_text = ax.set_ylabel('回归系数')
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()

    # #############################预测乐高玩具套装的价格###################################################


def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """从页面中读取数据，生成retX，retY列表（inFile：HTML文件 yr：年份 numPce：乐高部件数目 origPrc：原价）"""
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while len(currentRow) != 0:
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)


def setDataCollect(retX, retY):
    """依次读取六种乐高套装的数据，并生成数据矩阵"""
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)  # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)  # 2009年的乐高10196,部件数目3263,原价249.99


def useStandRegres():
    """使用线性回归进行测试"""
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    lgX1 = mat(ones((shape(lgX)[0], shape(lgX)[1] + 1)))    # 增加一个常数项的特征项X0
    lgX1[:, 1:shape(lgX1)[1] + 1] = mat(lgX)
    ws = standRegres(lgX1, lgY)
    print('%f %+f * 年份 %+f * 部件数量 %+f * 是否为全新 %+f * 原价' % (ws[0], ws[1], ws[2], ws[3], ws[4]))


def crossValidation(xArr, yArr, numVal=10):
    """交叉验证岭回归"""
    m = len(yArr)
    indexList = list(range(m))
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        # 创建训练集和测试集容器
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)    # 随机排序
        for j in range(m):
            # 将10%的数据划分为测试集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            # 用训练师的参数将测试数据标准化
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)    # 根据ws预测Y值（加Y的平均值是因为正则标准化后Y为0）
            errorMat[i, k] = rssError(yEst.T.A, array(testY))       # 统计误差
    meanErrors = mean(errorMat, 0)                                # 计算每次交叉验证的平均误差
    minMean = float(min(meanErrors))                              # 找到最小误差
    bestWeights = wMat[nonzero(meanErrors == minMean)]            # 找到最佳回归系数
    # 将标准化后的数据进行还原
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights/varX
    print('%f %+f * 年份 %+f * 部件数量 %+f * 是否为全新 %+f * 原价' % (
        (-1 * sum(multiply(meanX, unReg)) + mean(yMat)), unReg[0, 0], unReg[0, 1], unReg[0, 2], unReg[0, 3]))


if __name__ == '__main__':
    # plotRegression()
    # plotlwlrRegression()
    # plotabalonetest()
    # plotridgeRegres()
    # plotstageWiseMat()
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    useStandRegres()
    crossValidation(lgX, lgY)
    print(ridgeTest(lgX, lgY))