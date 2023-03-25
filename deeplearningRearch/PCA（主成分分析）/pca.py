import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 22835030 李昊
# 从文件读取数据
def load_csv(path):
    data_read = pd.read_csv(path)
    list = data_read.values.tolist()
    data = np.array(list)
    # print(data)
    return data
data = load_csv("test_score.csv")
# 输出列之间的相关系数矩阵
print("列之间的相关系数矩阵为：")
print(np.around(np.corrcoef(data.T),decimals=3))
# 对数据进行标准化
# St_data = StandardScaler().fit_transform(data)
St_data = (data - np.mean(data,axis=0))/np.std(data,axis=0)
print("标准化之后的矩阵为：")
print(St_data)
# 打印均值
print("标准化之后的矩阵均值为：")
print(np.mean(St_data))
# 打印方差
print("标准化之后的矩阵方差为：")
print(np.var(St_data))
# 进行主成分分析
# 计算样本的协方差矩阵p
covx = np.cov(St_data.T)
# 计算协方差矩阵的特征值和特征向量
featureValue,featureVector = np.linalg.eig(covx)
print("协方差矩阵的特征值和特征向量为：")
print(featureValue,featureVector)
# 对特征值及其特征向量从大到小排序
print("从大到小排序后协方差矩阵的特征值和特征向量为：")
featureValueOrder = np.argsort(featureValue)[::-1]
featureValueSort = featureValue[featureValueOrder]
featureVectorSort = featureVector[:,featureValueOrder]
print(featureValueSort)
print(featureVectorSort)
# 求特征值的贡献度
print("特征值的贡献度为：")
gx = featureValue/np.sum(featureValue)
print(gx)
# 选取第一主成分和第二主成分
#载荷矩阵
w1 = featureVectorSort[:,:1]
w2 = featureVectorSort[:,1:2]
print("主成分一的载荷矩阵为：")
print(w1)
print("主成分二的载荷矩阵为：")
print(w2)
# 主成分标准差
print("主成分的标准差为：")
print(np.std(w1))
print(np.std(w2))

# 读取6,7,45,30,49,26,44,8行数据
index = [6,7,45,30,49,26,44,8]
list = np.array([]);
for i in range(len(index)):
    list = np.append(list,data[index[i]-1,:],axis=0)
list = list.reshape(-1,6)
# 打印取出的数据
print(list)
# 对八个样本进行标准化
St_list = (list - np.mean(list,axis=0))/np.std(list,axis=0)
# 测试样本主成分取值
print("8个样本的第一主成分取值为：")
x_test = St_list.dot(w1)
print(x_test)
print("8个样本的第二主成分取值为：")
y_test = St_list.dot(w2)
print(y_test)
# 所有样本主成分取值
x_all = St_data.dot(w1)
y_all = St_data.dot(w2)
plt.scatter(x_all,y_all,label = "22835030 lihao")
plt.legend(loc='upper right')
plt.show()