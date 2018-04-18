# coding=utf-8
from numpy import *
import re
import operator
import feedparser


def loadDataSet():
    """预置一个词表集"""
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    # 1表示侮辱类，0表示不属于
    return postingList, classVec    # 词条切分后的分档和类别标签


def createVocabList(dataSet):
    """创建包含所有文档，但没有重复单词的list(单词库)"""
    vocabSet = set([])    # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)    # 创建两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """判断词汇表中的单词在输入文档中是否出现， 参数为词汇表和某个文档(朴素贝叶斯词集模型)"""
    returnVec = [0] * len(vocabList)    # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    # 如果单词存在则修改为1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec



def bagOfWords2VecMN(vocabList, inputSet):
    """朴素贝叶斯词袋模型"""
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    """朴素贝叶斯分类器训练函数， 参数为文档矩阵和每篇文档类别标签所构成的向量"""
    numTrainDoc = len(trainMatrix)     # 文档数目
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDoc)    # 文档中属于侮辱类的概率，等于1才能算，0是非侮辱类
    p0Num = ones(numWords)    # 初始化概率
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDoc):
        if trainCategory[i] == 1:    # 如果该行文档内有侮辱类词汇
            p1Num += trainMatrix[i]    # 向量相加（各位置相加）
            p1Denom += sum(trainMatrix[i])    # 数字相加，求总出现次数
        else:
            p0Num += trainMatrix[i]    # 向量相加（各位置相加）
            p0Denom += sum(trainMatrix[i])    # 数字相加，求总出现次数
    # p1Vect是[p(w1/c1),p(w2/c1)...p(wn/c1)],p0Vect是[p(w1/c0),p(w2/c0)...p(wn/c0)],p_abusive是p(c1)
    p1Vect = log(p1Num/p1Denom)    # 对每个元素做除法    此时自动将Denom转化为值为Denom的向量，进行向量除法
    p0Vect = log(p0Num/p0Denom)    # 用对数计算避免结果过小
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """朴素贝叶斯分类函数"""
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    # 元素相乘  加代表乘以非羞辱类概率。 没有除因为p(w)相同可以忽略
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """封装过后方便测试的便利函数"""
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:    # 生成文档对应词的矩阵 每个文档一行，每行内容为词向量
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))    # 根据每个词在文档中是否出现，生成包含词库内所有单词的向量
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))     # 判断测试词条在词汇list中是否出现，生成词向量
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))    # 将测试向量与返回概率相乘，输出结果
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):
    """将输入的文档除去小于两个字母的单词，并转化为小写单词集(文件解析)"""
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    """完整的垃圾邮件测试函数"""
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):    # 将两个文件夹下各25共50个邮件导入，并解析为词列表
        with open('email/spam/%d.txt' % i) as f:
            wordList = textParse(f.read())
            docList.append(wordList)    # append接受一个对象参数，将对象添加到列表的尾部
            fullText.extend(wordList)    # extend接受一个列表参数，并将其中的元素添加到列表的尾部
            classList.append(1)
        with open('email/ham/%d.txt' % i) as f:    # ham文件夹中23.txt第二行将SciFinance后的错误符号删掉即可
            wordList = textParse(f.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
    vocabList = createVocabList(docList)    # 将词列表中的重复删除，制成单词库
    trainingSet = list(range(50))    # 共50个数据集
    testSet = []
    for i in range(10):    # 通过随机数将10个数据集取出，作为测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:    # 进行训练
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))    # 将邮件生成词集模型并添加到训练数据矩阵
        trainClasses.append(classList[docIndex])    # 获取该邮件并添加到训练类别矩阵
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))    # 训练
    errorCount = 0
    for docIndex in testSet:    # 进行测试
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])    # 将邮件生成词集模型并添加到测试数据矩阵
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:    # 将分类结果与邮件类型比对，分析结果
            errorCount += 1
            print("classification error :", docList[docIndex])
    print("the error rate is: ", float(errorCount)/len(testSet))    # 计算错误率


def calcMostFreq(vocabList, fullText):
    """对所有词出现频率进行排序，返回排序后出现频率最高的前30个"""
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    """将两个RSS源作为参数,与spamTest()函数差别不大"""
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:    # 移除出现频率最高的30个单词
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(int(len(trainingSet)/5)):    # 选取的RSS源文章过少，因此视文章数量决定，若网络问题则依然会报错
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del[trainingSet[randIndex]]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append((bagOfWords2VecMN(vocabList, docList[docIndex])))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error :", docList[docIndex])
    print("the error rate is: ", float(errorCount)/len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    """最具表征性的词汇显示函数"""
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))    # 将出现概率过低的单词过滤
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)    # 按计算出的特征值进行排序
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])    # 按排序输出词表内的所有词汇
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


if __name__ =='__main__':
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # print(setOfWords2Vec(myVocabList, listOPosts[3]))
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print(trainMat)
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(p0V, "\n", p1V, pAb)
    # testingNB()
    # spamTest()
    # ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    # sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')    # 因网络问题访问不到书中给的网址
    sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')       # 从网上找到一个相对可行的替代品
    localWords(ny, sf)
    getTopWords(ny, sf)