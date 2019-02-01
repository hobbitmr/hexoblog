---
title: LogisticRegression的公式推导以及代码实现
date: 2018-12-11 14:14:14
tags:
categories:
- 机器学习
---
### 1.什么是逻辑回归。
逻辑回归是一种二分类的线性分类模型。通过已知样本，得到能够最佳拟合已知样本的函数，进而对未知样本进行分类。
### 2.逻辑回归的公式
我们假设 存在样本集$S= \{ (x^1,y^1),(x^2,y^2),(x^3,y^3)...(x^m,y^m) \} $ m个样本.$y_i$的取值是0或者1,$x_i$只有一个特征. 
那么得到训练集 $X=\{x^1,x^2,x^3,\cdots,x^m\},Y =\{y^1,y^2,\cdots,y^m\} $
我们先来看看线性回归的一般式:   
$\hat y^i=z(x^i) =\theta_0+\theta_1x^i $   
我们现在需要的有一个函数$h(x_i)$，能够输出(0,1) 的值。我们知道sigmod函数的输出值得范围刚好是(0,1)之间，所以我们在$z(x_i)$外面加一层sigmod函数:   
$h(x^i)= \frac{1}{1+e^{-z^i}}$   
这样子，我们就能够通过$h$函数去预测$y_i$的值.   
那么他的损失函数要如何定义: 
我们知道$\hat y $范围为[0,1]之间,如果我们把这个当成概率的话，那么$\hat y^i=1$的概率就为$h(x^i)$,$\hat y^i=0$的概率就为$1-h(x^i)$,表示如下：   
$ \begin{cases}
P(\hat y^i=1|x^i) = h(x^i)\\
P(\hat y^i=0|x^i) = 1-h(x^i)
\end{cases}$   
当$y^i$真实标记等于1的时候，我们是希望$\hat y^i$也尽量接近1，这样子，我们的损失函数才能最小.那么损失函数可以定义为：   
$ \begin{cases}
y^i=1:l(x^i) = -\ln(h(x^i))\\
y^i=0:l(x^i) = -\ln(1-h(x^i))
\end{cases}$   
我们可以通过下面函数图看看上面公式是否有效（以下图片来自吴恩达cs229机器学习视频）   
$l(x^i) = -\ln(1-h(x^i))$
![](/images/logistic_1.jpg)
$l(x^i) = -\ln(h(x^i)$
![](/images/logistic_2.jpg)
我们将两个函数合成一个函数有：   
$l(x^i) = -y^i\ln (h(x^i)-(1-y^i)\ln(1-h(x^i))$   
以上就是我们最终的损失函数。那么我们来定义以下cost函数 $j(x)$   
$j(\theta_0,\theta_1) = \frac{1}{m} \sum^m_{i=1}l(x^i)$
$j(\theta_0,\theta_1) = \frac{1}{m} \sum^m_{i=1}(-y^i\ln(h(x^i))-(1-y^i)\ln(1-h(x^i)))$

### 3.单特征梯度下降
$\frac{\partial h}{\partial z}=(\frac {1}{1+e^{-z}})'= \frac{e^{-z}}{(1+e^{-z})^2}=\frac{1+e^{-z}-1}{(1+e^{-z})^2}=(1-h(x^i))h(x^i)$   
$\frac{\partial l}{\partial h}=(-y^i\ln(h(x^i))-(1-y^i)\ln(1-h(x^i)))' =-\frac{y^i}{h(x^i)}+\frac{1-y^i}{1-h(x^i)}=\frac {h(x^i)-y^i}{h(x^i) * (1-h(x^i))}$ 
$dz =\frac{\partial l}{\partial z} = \frac{\partial l}{\partial h} * \frac{\partial h}{\partial z} =h(x^i)-y^i$
对$\theta_0$ 求导公式:   
$d\theta_0 =\frac{\partial j}{\partial \theta_0} = \frac{\partial z}{\partial \theta_0} * \frac{\partial h}{\partial z} * \frac{\partial l}{\partial h} * \frac{\partial j}{\partial l} =\frac{1}{m} \sum^m_{i=1}dz * \frac{\partial z}{\partial \theta_0}$
$d\theta_0 =\frac{1}{m} \sum^m_{i=1}(h(x^i)-y^i)$  
对$\theta_1$ 求导公式:   
$d\theta_1 =\frac{\partial j}{\partial \theta_1} = \frac{\partial z}{\partial \theta_1} * \frac{\partial h}{\partial z} * \frac{\partial l}{\partial h} * \frac{\partial j}{\partial l} =\frac{1}{m} \sum^m_{i=1} dz * \frac{\partial z}{\partial \theta_1}$   
$d\theta_1 = x^i * dz =\frac{1}{m} \sum^m_{i=1}x^i*(h(x^i)-y^i)$   
$\theta_0 = \theta_0 -a * d\theta_0$   
$\theta_1 = \theta_1 -a * d\theta_1$

### 4.多特征梯度下降
 假设 $x_i$的n个特征，那么得到$x^i=\{x_1^i,x_1^i,x_1^i,...,x_n^i\}$的列向量 
 那么得到矩阵
 $X = \begin{bmatrix}
{x_1^1}&{x_1^2}&{\cdots}&{x_1^m}\\
{x_2^1}&{x_2^2}&{\cdots}&{x_2^m}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{x_n^1}&{x_n^2}&{\cdots}&{x_n^m}
\end{bmatrix}$ 维度为(n,m)
 假设 $\theta=\{\theta_0,\theta_1,\theta_2,...,\theta_n\}$的列向量,维度为(n+1)   
那么由此得到:   
$\hat y^i=z(x^i) =\theta_0+\theta_1x_1^i+\theta_2x_2^i+\theta_3x_3^i+...+\theta_nx_n^i$
$h(x^i)= \frac{1}{1+e^{-z(x^i)}}$   
 $j(\theta_0,\theta_1,\cdots,\theta_n) = \frac{1}{m} \sum^m_{i=1}(-y^i\ln(h(x^i))-(1-y^i)\ln(1-h(x^i)))$
先求出函数$h$对函数$l$的导数$dz^i={h(x^i)-y^i}$  
 对 $\theta$ 求导:   
 ${dJ\over d\theta_0} = {1\over m}\sum_{i=1}^m {dl\over dz^i} * {dz^i\over d\theta_0} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_0^i$   
 ${dJ\over d\theta_1} = {1\over m}\sum_{i=1}^m {dl\over dz^i} * {dz^i\over d\theta_1} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_1^i$
$\vdots$
  ${dJ\over d\theta_n} = {1\over m}\sum_{i=1}^m {dl\over dz^i} * {dz^i\over d\theta_n} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_n^i$ 
最后进行梯度下降运算
 $\theta_0 = \theta_0-a * {dJ\over d\theta_0}$  
 $\theta_1 = \theta_1-a * {dJ\over d\theta_1}$  
 $\vdots$
 $\theta_n = \theta_n-a * {dJ\over d\theta_n}$
 
### 5.添加正则化
增加正则项的目的是对$\theta$进行惩，防止过拟合。公式如下：   
$j(\theta_0,\theta_1,\cdots,\theta_n) = \frac{1}{m} \sum^m_{i=1}(-y^i\ln(h(x^i))-(1-y^i)\ln(1-h(x^i)))+{\lambda \over 2m}\sum_{i=1}^n\theta_i^2$ 
正则项求导(i>=1)   
$dRegu(\theta_i)={\lambda \over m}\theta_i$
添加了正则项后对$\theta$求导。$\theta_0$项不用添加正则约束：   
 ${dJ\over d\theta_0} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i) * x_0^i$   
 ${dJ\over d\theta_1} =  {1\over m}\sum_{i=1}^m [(h(x^i)-y^i)* x_1^i+\lambda\theta_1]$
$\vdots$
  ${dJ\over d\theta_n}=  {1\over m}\sum_{i=1}^m [(h(x^i)-y^i)* x_n^i+\lambda\theta_n]$ 
### 6.向量化运算:
为了提高运算效率，我们将上面的公式进行向量化运算。
我们将X的特征扩展1个维度，第0维的数据为1,得到矩阵如下
 $X = \begin{bmatrix}
{1}&{1}&{\cdots}&{1}\\
{x_1^1}&{x_1^2}&{\cdots}&{x_1^m}\\
{x_2^1}&{x_2^2}&{\cdots}&{x_2^m}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{x_n^1}&{x_n^2}&{\cdots}&{x_n^m}
\end{bmatrix}$ 维度为(n+1,m)
假设 $\theta=\{\theta_0,\theta_1,\theta_2,...,\theta_n\}$的列向量,维度为(n+1)  
得到:   
$\hat y^i=z(x^i) =\theta_0x_0^i+\theta_1x_1^i+\theta_2x_2^i+\theta_3x_3^i+...+\theta_nx_n^i=\theta^Tx^i$
$h(x^i)= \frac{1}{1+e^{-z(x^i)}}$    
得到矩阵:$H = [h(x^1),h(x^2),\cdots,h(x^m)] = h(\theta^TX)$
 $j(\theta_0,\theta_1,\cdots,\theta_n) = \frac{1}{m} \sum^m_{i=1}(-y^i\ln(h(x^i))-(1-y^i)\ln(1-h(x^i)))$   
 代价函数为：$J(\theta) = \frac {1}{m} * (-\ln(H)Y^T-\ln(1-H)(1-Y)^T)$  
反向求导:
将 $dz^i={h(x^i)-y^i}$ 
$dZ=[dz^1,dz^2,\cdots,dz^i] = H-Y$ 

正则化求导向量化
$dR=
\begin{bmatrix}
0\\
dRegu(\theta_1)\\
dRegu(\theta_2)\\
\vdots\\
dRegu(\theta_n)
\end{bmatrix}=
\begin{bmatrix}
0\\
{\lambda \over m}\theta_1\\
{\lambda \over m}\theta_2\\
\vdots\\
{\lambda \over m}\theta_n
\end{bmatrix}
$

对$\theta$ 求导公式(\* 号表示矩阵点乘):  
$d\theta = \begin{bmatrix}
{1\over m}\sum_{i=1}^m {dl\over dz^i} * {dz^i\over d\theta_1} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_0^i\\
{1\over m}\sum_{i=1}^m {dl\over dz^i} * {dz^i\over d\theta_1} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_1^i\\
\vdots\\
{1\over m}\sum_{i=1}^m {dl\over dz^i} * {dz^i\over d\theta_1} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_n^i
\end{bmatrix}
=\frac{1}{m}X(H-Y)^T+dR$   
$\theta =\theta -a * d\theta $

### 7.代码实现

``` python
def sigmoid(fz):
    fh = 1 / (1 + np.exp(-fz))
    return fh
 theta = None
    def fit(self,input_x,input_y,leaning_rate=0.001,_lambda=0):
        input_x = np.column_stack([np.ones([input_x.shape[0],1]),input_x])
        input_x = np.transpose(input_x)
        input_y = np.reshape(input_y,[1,-1])
        if self.theta is None:
            self.theta = np.random.random([input_x.shape[0],1])
        # options = {'maxiter': 400}
        # res = optimize.minimize(LogisticRegression.costFunction,
        #                         self.theta,
        #                         (input_x, input_y),
        #                         jac=True,
        #                         method='Powell',
        #                         options=options)
        # cost = res.fun
        # self.theta = res.x

        theta,cost  = CustomOptimize.gradientDescent(LogisticRegression.costFunction,
                                self.theta,
                                (input_x, input_y),leaning_rate=leaning_rate,_lambda=_lambda)
        self.theta=theta
        return cost



    def predict(self, input_x):
        input_x = np.column_stack([np.ones([input_x.shape[0], 1]), input_x])
        input_x = np.transpose(input_x)
        fz = np.dot(np.transpose(self.theta), input_x)
        return  np.transpose(sigmoid(fz))

    def costFunction(theta, input_x, input_y,_lambda=0):
        m = input_x.shape[1]
        theta = np.reshape(theta, [-1, 1])
        fz = np.dot(np.transpose(theta), input_x)
        fh = sigmoid(fz)
        dz = fh - input_y
        J = (1.0 / m) * np.squeeze(
            -np.dot(np.log(fh), np.transpose(input_y)) - np.dot(np.log(1 - fh), np.transpose(1 - input_y)))+(_lambda/(2*m))*np.sum(np.power(theta,2))
        J = np.squeeze(J)

        # 添加正则化
        dr = np.zeros([theta.shape[0], 1])
        dr[1:] = theta[1:]
        dr = (_lambda / m) * dr

        d_theta = (1.0 / m) * np.dot(input_x, np.transpose(dz))+dr
        d_theta = np.squeeze(d_theta)
        return J, d_theta

class CustomOptimize:
    def gradientDescent(costFunction,theta, data=(),leaning_rate=0.01,_lambda=0):
        """
        :param costFunction:传入一个costFunction,要求该函数的参数为：costFunction(theta, input_x, input_y)，
        返回return cost(损失函数), d_theta(m,) (theta的导数)
        :param theta: 初始化的theta
        :param data: 参数是(X,y)的形式
        :param leaning_rate:
        :return: cost(损失函数), theta(更新后的theta)
        """
        X, y = data
        cost, d_theta = costFunction(theta, X, y,_lambda)
        d_theta = np.reshape(d_theta,[-1,1])
        theta = theta - leaning_rate * d_theta
        return theta,cost
```
运行测试:
``` python
import numpy as np
from LinearModel import LogisticRegression
import matplotlib.pyplot as plt
import os
import FeatureUtil




data = np.loadtxt(os.path.join( 'week2/Data/ex2data1.txt'), delimiter=',')
xdata, ydata = data[:, 0:-1], data[:,-1]
xdata=FeatureUtil.featureNormalize(xdata)
# xdata1 =np.power(xdata,2)


clf = LogisticRegression()

plt.figure(1)
plt.ion()
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.title("cs229 dataset")
pos_1 = ydata ==1
pos_2 = ydata ==0
plt.scatter(xdata[pos_1,0],xdata[pos_1,1] , marker="*", c="k")
plt.scatter(xdata[pos_2,0],xdata[pos_2,1] , marker="o", c="y")
plt.show()
line=None
for i in range(10000):
    coss=clf.fit(xdata,ydata,leaning_rate=0.1)
    theta=clf.theta
    x2 = -(theta[0]+theta[1]*xdata[:,0:1])/theta[2]
    #x2=np.sqrt(x2)
    if line is None:
        line,= plt.plot(xdata[:,0:1], x2, color='g')
    line.set_ydata(x2)
    print("coss:", coss)
    plt.pause(0.1)

    #画函数图

    #line.set_ydata(clf.predict(xdata))
    # plt.pause(0.1)
    # print("coss:", coss)

```
 代码参考[github](https://github.com/hobbitmr/deeplearnPractice/tree/master/cs229)
 运行效果：
 ![avatar](/images/logistic_3.png)













   