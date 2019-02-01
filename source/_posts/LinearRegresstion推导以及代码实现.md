---
title: LinearRegresstion推导以及代码实现
date: 2018-11-29 17:33:19
mathjax: true
categories:
- 机器学习
---
有一个场景是这样，根据已知数据集，去预测房价的走向，这个是属于回归问题，所以我们使用线性回归来解决这个问题。我们先来看看线性回归中最简单的形式。
 ### 1.线性回归公式
 假设输入为$X=\{x^1,x^2,x^3,\cdots,x^m\},Y =\{y^1,y^2,\cdots,y^m\} $ m个样本，为了简单，我们先假设每个样本只有一个特征。为则函数模型为
 $h(x^i)= \theta_0+\theta_1x^i$
 我们采用最小乘二法作为损失函数：
 $l(x^i)=  (h(x^i)-y^i)^2$
 那么成本函数，就是所有样本的损失函数加起来，然后平均值。
 $J(\theta_1,\theta_0) ={1\over 2m}\sum_{i=1}^m(h(x^i)-y^i)^2$ 
 那么可以看到，如果我们预测的值跟实际值越相近，那么$J$的值就越小，忧郁$J$是凸函数，所以我们需要找到$J$函数的最小值，如果当函数导数为0的时候，就是$J$函数的最小值，所以我们可以通过梯度下降法，求得最佳的$\theta_1$和$\theta_0$，取得最小的$J$函数的最小值.
 ### 2.单特征梯度下降
先求出函数$h$对函数$l$的导数$dh^i={2(h(x^i)-y^i)}$   
 对 $\theta_1$ 求导:   
 ${dJ\over d\theta_1} = {1\over 2m}\sum_{i=1}^m {dl\over dh^i} * {dh^i\over d\theta_1} $   
 ${dJ\over d\theta_1} = {1\over 2m}\sum_{i=1}^m dh^i * x^i $  
 ${dJ\over d\theta_1} = {1\over m}\sum_{i=1}^m {(h^i-y^i)} * x^i $   
 对 $\theta_0$ 求导：
 ${dJ\over d\theta_0} = {1\over 2m}\sum_{i=1}^m {dl\over dh^i} * {dh^i\over d\theta_0} $   
 ${dJ\over d\theta_0} = {1\over 2m}\sum_{i=1}^m dh_i$   
 ${dJ\over d\theta_0} = {1\over m}\sum_{i=1}^m {(h^i-y^i)}$   
 然后进行梯度下降  
 $\theta_1 = \theta_1-a * {dJ\over d\theta_1}$  
 $\theta_0 = \theta_0-a * {dJ\over d\theta_0}$
### 2.多特征梯度下降
 上面假设x只有一个特征，但是实际上数据肯定是多维度的，有多个属性。所以我们对上面的公式做个简单的修改. 
 假设 $x_i$的维度n，那么得到$x^i=\{x_1^i,x_1^i,x_1^i,...,x_n^i\}$的列向量 
 那么得到矩阵
 $X = \begin{bmatrix}
{x_1^1}&{x_1^2}&{\cdots}&{x_1^m}\\
{x_2^1}&{x_2^2}&{\cdots}&{x_2^m}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{x_n^1}&{x_n^2}&{\cdots}&{x_n^m}
\end{bmatrix}$ 维度为(n,m)
 假设 $\theta=\{\theta_0,\theta_1,\theta_2,...,\theta_n\}$的列向量,维度为(n+1)   
那么由此得到:
 $h(x^i)= \theta_0+\theta_1x_1^i+\theta_2x_2^i+\theta_3x_3^i+...+\theta_nx_n^i$ 
 $l(x^i)=  (h(x^i)-y^i)^2$
 $J(\theta_0,...,\theta_n)={1\over 2m}\sum_{i=1}^m(h(x^i)-y^i)^2$   
先求出函数$h$对函数$l$的导数$dh^i={2(h(x^i)-y^i)}$
 对 $\theta$ 求导:   
 ${dJ\over d\theta_0} = {1\over 2m}\sum_{i=1}^m {dl\over dh^i} * {dh^i\over d\theta_0} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i) * x_0^i$   
 ${dJ\over d\theta_1} = {1\over 2m}\sum_{i=1}^m {dl\over dh^i} * {dh^i\over d\theta_1} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_1^i$
$\vdots$
  ${dJ\over d\theta_n} = {1\over 2m}\sum_{i=1}^m {dl\over dh^i} * {dh^i\over d\theta_n} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_n^i$ 
最后进行梯度下降运算
 $\theta_0 = \theta_0-a * {dJ\over d\theta_0}$  
 $\theta_1 = \theta_1-a * {dJ\over d\theta_1}$  
 $\vdots$
 $\theta_n = \theta_n-a * {dJ\over d\theta_n}$

### 3.增加正则项
增加正则项的目的是对$\theta$进行惩，防止过拟合。公式如下：   
$J(\theta_0,\theta_1,\cdots,\theta_n) ={1\over 2m}\sum_{i=1}^m(h(x^i)-y^i)^2+{\lambda \over 2m}\sum_{i=1}^n\theta_i^2$ 
正则项求导(i>=1)   
$dRegu(\theta_i)={\lambda \over m}\theta_i$
添加了正则项后对$\theta$求导。$\theta_0$项不用添加正则约束：   
 ${dJ\over d\theta_0} =  {1\over m}\sum_{i=1}^m (h(x^i)-y^i) * x_0^i$   
 ${dJ\over d\theta_1} =  {1\over m}\sum_{i=1}^m [(h(x^i)-y^i)* x_1^i+\lambda\theta_1]$
$\vdots$
  ${dJ\over d\theta_n}=  {1\over m}\sum_{i=1}^m [(h(x^i)-y^i)* x_n^i+\lambda\theta_n]$ 
### 4.向量化运算:
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
那么由此得到:
 $h(x^i)= \theta_0x_0^i+\theta_1x_1^i+\theta_2x_2^i+\theta_3x_3^i+...+\theta_nx_n^i = \theta^T x^i $
那么得到矩阵$H(X) =[h(x^1),h(x^2)...h(x^m)] = \theta^TX$ 维度为（1,m）
 $l(x^i)=  (h(x^i)-y^i)^2$
 $L=[l(x^1),l(x^2),\cdots,l(x^m)]=  (H-Y)^2$ 维度为(1,m)
 $J(\theta)={1\over 2m} * (H-Y)(H-Y)^T+{\lambda \over 2m}\sum_{i=1}^n\theta_i^2$
反向求导:   
$dh^i={2(h(x^i)-y^i)}$ 
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
对 $\theta$ 求导:  
 $d\theta=
 \begin{bmatrix}
{dJ\over d\theta_0}\\
{dJ\over d\theta_1}\\
\vdots\\
{dJ\over d\theta_n}
 \end{bmatrix}
 =\begin{bmatrix}
 {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_0^i+0\\
 {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_1^i+{\lambda \over m}\theta_1\\
 \vdots\\
 {1\over m}\sum_{i=1}^m (h(x^i)-y^i)* x_n^i+{\lambda \over m}\theta_n
 \end{bmatrix}= {1\over m} * (X(H-Y)^T+dR) $   
 梯度下降更新:
 $\theta:=\theta - a * d\theta $

### 5.代码实现:
终于推导完全，接下来我们看看代码怎么实现吧
``` python
def sigmoid(fz):
    fh = 1 / (1 + np.exp(-fz))
    return fh
class LinearRegression:
    theta =None
    def fit(self,input_x,input_y,leaning_rate=0.001,_lambda=0):
        input_x = np.column_stack([np.ones([input_x.shape[0],1]),input_x])
        input_x = np.transpose(input_x)
        input_y = np.reshape(input_y,[1,-1])
        if self.theta is None:
            self.theta = np.random.random([input_x.shape[0],1])
        options = {'maxiter': 400}
        # res = optimize.minimize(LinearRegression.costFunction,
        #                         self.theta,
        #                         (input_x, input_y),
        #                         jac=True,
        #                         method='Powell',
        #                         options=options)
        # cost = res.fun
        # self.theta = res.x

        theta,cost  = CustomOptimize.gradientDescent(LinearRegression.costFunction,
                                self.theta,
                                (input_x, input_y),leaning_rate=leaning_rate,_lambda=_lambda)
        self.theta=theta
        return cost

    def costFunction(theta, input_x, input_y,_lambda=0):
        m = input_x.shape[1]
        theta = np.reshape(theta, [-1, 1])
        # 对应的公式 Z=theta*X
        fz = np.dot(np.transpose(theta), input_x)
        J = np.matmul((fz-input_y),np.transpose(fz-input_y))/ (2*m)+(_lambda/(2*m))*np.sum(np.power(theta,2))
        J = np.squeeze(J)
        dh = 2.0*(fz-input_y)

        #添加正则化
        dr=np.zeros([theta.shape[0],1])
        dr[1:] = theta[1:]
        dr = (_lambda/m) *dr
        d_theta = (1.0 / m) * np.dot(input_x, np.transpose(dh))+dr
        d_theta = np.squeeze(d_theta)
        return J, d_theta

    def predict(self, input_x):
        input_x = np.column_stack([np.ones([input_x.shape[0], 1]), input_x])
        input_x = np.transpose(input_x)
        fh = np.dot(np.transpose(self.theta), input_x)
        return np.transpose(fh)
        
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
from   sklearn import datasets
from LinearModel import LinearRegression
import matplotlib.pyplot as plt
import os
import FeatureUtil

#加载数据集
# irisdata =  datasets.load_iris()
# # 我们这里只使用一个特征，方便画散点图。
# xdata = irisdata.data[:,2]
# xdata= np.reshape(xdata,newshape=(-1,1))
# ydata = irisdata.data[:,3]
# ydata= np.reshape(ydata,newshape=(-1,1))

data = np.loadtxt(os.path.join( 'week1/Data/ex1data1.txt'), delimiter=',')
xdata, ydata = data[:, 0:-1], data[:,1]

#xdata=FeatureUtil.featureNormalize(xdata)
ydata= np.reshape(ydata,newshape=(-1,1))


clf = LinearRegression()

plt.figure(1)
plt.ion()
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.title("iris dataset")
plt.scatter(xdata,ydata , marker="o", c="r")
line=None
plt.show()
for i in range(2000):
    coss=clf.fit(xdata,ydata,leaning_rate=0.01)
    #画函数图
    if line is None:
        line, = plt.plot(xdata, clf.predict(xdata), color='g')
    line.set_ydata(clf.predict(xdata))
    plt.pause(0.1)
    print("coss:", coss)
```
 代码参考[github](https://github.com/hobbitmr/deeplearnPractice/tree/master/cs229)
 运行效果：
 ![avatar](/images/LinearRegresstion.png)
 
 
 
 
 



