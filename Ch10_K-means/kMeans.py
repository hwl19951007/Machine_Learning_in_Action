from numpy import *
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """便利函数，从文件中读取数据并封装到矩阵中"""
    dataMat = []
    with open(fileName) as f:
        for line in f.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))
            dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    """计算两个向量的欧氏距离"""
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """为给定数据集构建一个包含k个随机质心的集合"""
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))    # 初始化质心集合
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)    # 找出各个特征的最小值最大值及范围
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)    # 在最小值和最大值之间随机一个值作为质心
    return centroids


def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    """K-means算法，输入：数据集，质心数，距离算法，初始化质心点算法，输出：质心点，分类结果"""
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))    # 簇分配结果矩阵，第一列记录簇索引值，第二列记录存储误差（当前点到簇质心的距离）
    centroids = createCent(dataSet, k)    # 初始化质心
    clusterChanged = True    # 标志变量，当有簇分类结果变化时为True
    while clusterChanged:    # 当有质心变化时
        clusterChanged = False
        for i in range(m):    # 遍历每个数据点
            minDist = inf
            minIndex = -1
            for j in range(k):    # 遍历每个质心点，计算与质心的距离，选择最优质心
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):    # 遍历每个质心点，更新质心
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]    # 通过数组过滤获取当前簇的数据点
            centroids[cent, :] = mean(ptsInClust, axis=0)    # 更新质心，axis=0 表示沿列方向进行均值计算
    return centroids, clusterAssment


# ###################################使用后处理来提高聚类性能之二分K-means算法#########################################


def biKmeans(dataSet, k, distMeans=distEclud):
    """二分K-means算法：不断选择使误差最低化的簇进行二分直到拥有k个簇"""
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))    # 簇分配结果矩阵，第一列记录簇索引值，第二列记录存储误差（当前点到簇质心的距离）
    centroid0 = mean(dataSet, axis=0).tolist()[0]    # 计算整个数据集的质心
    centList = [centroid0]                           # 用一个列表来保留所有的质心
    for j in range(m):                               # 计算误差平方和
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:                         # 当质心数量还不够时
        lowestSSE = inf                              # 初始化最小误差平方和
        for i in range(len(centList)):               # 遍历当前已有的质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]    # 通过数组过滤获取当前簇的数据点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)      # 对当前簇进行K-means分为两簇
            sseSplit = sum(splitClustAss[:, 1])                                      # 划分后者两簇的误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])    # 其他簇的误差平方和
            print("簇划分后这两簇的误差平方和以及其他簇的误差平方和： ", sseSplit, sseNotSplit)
            if sseSplit + sseNotSplit < lowestSSE:                       # 如果整个数据集的误差平方和减小则本次划分保存
                bestCentToSplit = i                                      # 最佳划分簇
                bestNewCents = centroidMat                               # 簇划分后的两个新质心
                bestClustAss = splitClustAss.copy()                      # 簇划分后的数据集归属质心及误差平方和
                lowestSSE = sseSplit + sseNotSplit                       # 簇划分后的总最小误差平方和
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)    # 将其中一簇质心索引值设置为最后当前最后一个
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 将另一簇质心索引值设置为被划分的簇质心索引值
        print("最适合划分的簇的序号为： ", bestCentToSplit)
        print("被重新划分的数据点有:  ", len(bestClustAss), "个")
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]    # 将划分后的第一个质心存储到当前簇质心索引值
        centList.append(bestNewCents[1, :].tolist()[0])               # 将将划分后的第二个质心添加到centList队列的最后
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # 将被划分簇的质心更新为划分后的质心
    return mat(centList), clusterAssment


# ###########################################对地图上的点进行聚类#####################################################


def distSLC(vecA, vecB):
    """返回地球表面两点间距离，单位是英里。通过余弦定理进行计算"""
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0


def clsterClubs(numClust=5):
    """簇绘图，将地图上的点进行聚类并绘制"""
    dataList = []                                                                # 初始化坐标列表
    with open('places.txt') as f:
        for line in f.readlines():
            lineArr = line.split('\t')
            dataList.append([float(lineArr[4]), float(lineArr[3])])              # 获取地址经纬度
    dataMat = mat(dataList)                                                      # 转化为坐标矩阵
    myCentroids, clustAssing = biKmeans(dataMat, numClust, distMeans=distSLC)    # 聚类出要求的质心数
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]                                                  # 创建矩阵
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']          # 创建不同的标识图案
    axprops = dict(xticks=[], yticks=[])                                         # 用于控制边框上的数字刻度
    ax0 = fig.add_axes(rect, label='ax0', **axprops)                             # 创建一个矩阵
    imgP = plt.imread('portland.png')                                            # 基于一幅图像来创建矩阵
    ax0.imshow(imgP)                                                             # 绘制该矩阵
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)                         # frameon控制是否显示边框
    for i in range(numClust):
        ptsInCurrCluster = dataMat[nonzero(clustAssing[:, 0].A == i)[0], :]      # 获取该聚类内的地点
        markerStyle = scatterMarkers[i % len(scatterMarkers)]                    # 选择该聚类的标识图案，并绘制地点及质心
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
        ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()







if __name__ == "__main__":
    # dataMat = mat(loadDataSet('testSet.txt'))
    # myCentroids, clustAssing = kMeans(dataMat, 4)
    # print(myCentroids)
    dataMat3 = mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(dataMat3, 4)
    print(centList)
    clsterClubs()
