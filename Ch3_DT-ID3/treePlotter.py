import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")    # 定义文本框和箭头格式
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """绘制箭头的注解"""
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):
    """获取叶节点的数目"""
    numLeafs = 0
    firstSides = list(myTree.keys())    # 需将字典转化为list后才能提取键值
    firstStr = firstSides[0]    # 找到第一个元素，为当前分类标签
    secondDict = myTree[firstStr]
    for key in secondDict.keys():    # 通过递归获取叶节点的数目
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """获取决策树的层数"""
    maxDepth = 0
    thisDepth = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():    # 通过递归获取当前树的层数
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        elif thisDepth > 1:    # 原代码中，如果最后一个节点是叶子节点，则决策树层数不会被正常计算，因此增加了一层判断来修正逻辑
            break
        else:
            thisDepth = 1
    if thisDepth > maxDepth:
        maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    """输出预先存储的树信息，避免每次测试都需要重新创建树"""
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    """在父子节点间填充文本信息"""
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]    # 计算父子节点的中点位置
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    """绘制决策树。xOff和yOff用于记录当前绘制的叶子节点的位置"""
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    # cntrPt用来记录当前要画的树的树根的结点位置，横竖坐标在主函数中有注释
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)    # 标注节点信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)    # 箭头内注解
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD    # 初始值为1，每次绘制新的节点，减少总层数分之一
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):    # 如果下个节点仍是分支，递归绘制决策树
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW    # 如果是叶子节点，向右总宽度分之一进行绘制
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    """主函数，绘制决策树图"""
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # plotTree.xOff    为使节点绘制的位置位于整体宽度中间，使节点初始值为总层数的负二分之一。每次添加总层数分之一
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0    # 初始值为1，每次绘制新的节点，减少总层数分之一
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()





myTree = retrieveTree(0)
# print(getNumLeafs(myTree))
# print(getTreeDepth(myTree))
print(myTree)
myTree['no surfacing'][3] = 'maybe'
print(myTree)
# print(getNumLeafs(myTree))
# print(getTreeDepth(myTree))
createPlot(myTree)













