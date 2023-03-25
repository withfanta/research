import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
# 定义目标函数
def func(x):
    return np.sin(x) + 0.3 * x
#生成一个拟合的多项式
def fit_func(p,x):
    f = np.poly1d(p)
    return f(x)
#定义误差
def error(p,x,y):
    err = fit_func(p,x) - y
    return err
#生成100个点
x_points = np.linspace(0,10,1000)
x = x_points[::10]
x_train = x[::2]
x_test = x[1::2]
y_train = func(x_train)
y_bias_train = y_train + np.random.normal(0,0.2,50)
print(y_bias_train)
plt.plot(x_points,func(x_points),color='r',label="Target funtion")
plt.scatter(x_train,y_bias_train,color='b',label="Training samples")
#用最小二乘法拟合
# M是这个多项式的次数
def fit(M=0):
    # 生成M+1项作为多项式的参数
    p_init = np.random.rand(M+1)
    p_res = leastsq(error,p_init,args=(x_train,y_bias_train))
    print(p_res)
    #打印拟合后的函数图像，使用x_test作为点
    plt.plot(x_test, fit_func(p_res[0],x_test),color='g',label="Learned funtion")
    R2 = 1 - np.sum((y_train - fit_func(p_res[0], x_test)) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    print(R2)
fit(6)
plt.legend(loc='upper right')
plt.show()