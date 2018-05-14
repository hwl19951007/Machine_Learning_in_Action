from numpy import *
from tkinter import *
import regTrees

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tolS, tolN):
    """绘制图像"""
    reDraw.f.clf()    # 清空之前的图像，保证不会重叠
    reDraw.a = reDraw.f.add_subplot(111)    # 清空后需要重新添加新图
    if chkBtnVar.get():    # 检查复选框是否被选中，若选中则建立模型树并进行预测
        if tolN < 2:
            tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)
    else:    # 若未选中则建立回归树并进行预测
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)    # 绘制真实值，离散型散点图
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)    # 绘制预测值，连续曲线
    reDraw.canvas.show()


def getInputs():
    """理解输入并防止程序崩溃"""
    try:
        tolN = int(tolNentry.get())                        # 期望得到整型数
    except e:                                              # 若出错则提示错误并替换回默认值
        tolN = 10
        print("请输入tolN（切分的最少样本数，整数形式）")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:                                                   # 期望得到整型数
        tolS = float(tolSentry.get())
    except e:                                              # 若出错则提示错误并替换回默认值
        tolS = 1.0
        print("请输入tolS（容许的误差下降值）")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '10')
    return tolN, tolS


def drawNewTree():
    """获取输入框的值后调用reDraw 函数绘制树"""
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)


root = Tk()

reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)    # .grid()设定行和列的位置，rowspan和columnspan设置允许跨行或跨列
tolNentry = Entry(root)                                    # Entry 文本输入框
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Button(root, text="重新绘制", command=drawNewTree).grid(row=2, column=2, rowspan=3)    # Button 按钮

chkBtnVar = IntVar()                                      # IntVar 按键整数值 用于读取复选按钮的取值
chkBtn = Checkbutton(root, text="模型树", variable=chkBtnVar)    # Checkbutton 复选按钮
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)

root.mainloop()





