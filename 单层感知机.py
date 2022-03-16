# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/7 19:57
@Auth ： victory.He
@File ：TensorFlow.py
@IDE ：PyCharm
@Motto：(Always Be Coding)
@Function： 单层感知机器
"""
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

Tarin_data = [
    [1, 3, 1],
    [2, 5, 1],
    [3, 8, 1],
    [2, 6, 1],
    [3, 1, -1],
    [4, 1, -1],
    [6, 2, -1],
    [7, 3, -1]
]

Flage = True  # 用于标记是否需要继续训练
weight = [0, 0]  # 权重
# global bias
# bias = 0  # 与标签值的差值
learning_rate = 0.2  # 学习效率


def sin(value):  # 线性分类
    sig = 0
    if value > 0:
        sig = 1
    else:
        sig = -1
        pass
    return sig


# 训练
def Train():
    global bias # 与标签值的差值
    bias=0
    train_num = int(input("请输入训练次数:"))
    for i in range(train_num):
        x1, x2, y = random.choice(Tarin_data)  # x1 表示坐标 第一位  x2表示第二位 y 表示标签值
        print(x1, x2, y)
        y_predict = sin(weight[0] * x1 + weight[1] * x2)
        print("训练数据 :  x:(%d,%d) y:%d\t==>\t y_predict:\t%d" % (x1, x2, y, y_predict))
        if y * y_predict <= 0:      #判断 y真实值 与 预测值是否相同
            weight[0] = weight[0] + learning_rate * y * x1
            weight[1] = weight[1] + learning_rate * y * x2

            bias = bias + learning_rate * y
            print("更新\t权重\t和\t偏差:")
            print( weight[0] + weight[1] +bias)
        print("停止更新")
        print(weight[0], weight[1], bias)
        # weight.append(round(num,3))    #随机权重 保留三位小数
        pass
    return weight, bias


def date_loading(path):  # 导入本地csv文件
    try:
        df = pd.read_csv(path, header=None)
        df.tail()
    except Exception as Result:
        print("路径错误", Result)


def date_input():  # 数据输入
    data = []
    list_data = input("请输入一个数据数组:")
    print("输入 q 退出")
    if list_data == "q" or list_data == "Q":
        Flage = False
    data += [int(n) for n in list_data.split(",")]
    return data


def draw():
    if(len(Tarin_data)==8):
        plt.plot(np.array(Tarin_data)[0:3, 0], np.array(Tarin_data)[0:3, 1], 'ro')
        plt.plot(np.array(Tarin_data)[4:, 0], np.array(Tarin_data)[4:, 1], 'bo')
        x_1 = []
        x_2 = []
        for i in range(-10, 10):
            x_1.append(i)
            x_2.append((-weight[0] * i - bias) / weight[1])
        plt.plot(x_1, x_2)
        plt.show()
    else:
        lenth=len(Tarin_data)
        count=0
        for i in Tarin_data:
            if (i[2] == -1):
                count += 1
                pass
        plt.plot(np.array(Tarin_data)[0:lenth-count-1, 0], np.array(Tarin_data)[0:lenth-count-1, 1], 'ro')
        plt.plot(np.array(Tarin_data)[lenth-count:, 0], np.array(Tarin_data)[lenth-count:, 1], 'bo')
        x_1 = []
        x_2 = []
        for i in range(-10, 10):
            x_1.append(i)
            x_2.append((-weight[0] * i - bias) / weight[1])
        plt.plot(x_1, x_2)
        plt.show()


def check_point(data):
    if(data[2]==-1):
        Tarin_data.append(data)
        pass
    else:
        Tarin_data.insert(0,data)


def test():
    weight, bias = Train()
    draw()
    while (Flage):
        test_data = date_input()
        predict = sin(weight[0] * test_data[0] + weight[1] * test_data[1] + bias)
        list=[test_data[0],test_data[1],predict]
        check_point(list)
        print(Tarin_data)
        print("预测值 ：%d" % predict)
        draw()


if __name__ == "__main__":
    test()
