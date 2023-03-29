import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import os
import cv2
from pathlib import Path


# 把文件读为矩阵
def readAsMatrix(filename):
    data_read = pd.read_csv(filename, sep=' ', header=None,
                                   skiprows=1, names=['JD', 'Magnorm', 'Mage'])
    return data_read
target_dara_read =  readAsMatrix('ref_044_16280425-G0013_364820_9174')
normal_data_read1 = readAsMatrix('044_16280425-G0013/ref_044_16280425-G0013_331771_5921.txt')


# 遍历文件夹读为矩阵
def readFolderAsMatrix(foldername):
    filenames = os.listdir(foldername)
    martixList = []
    count = 0
    for filename in filenames:
        filePath = foldername+'/'+filename
        folder_data_read = pd.read_csv(filePath, sep=' ', header=None,skiprows=1, names=['JD', 'Magnorm', 'Mage'])
        martixList.append(folder_data_read)
        count += 1
    martixList_concat = pd.concat(martixList,axis=0,ignore_index=True)
    print(count)
    # print(martixList_concat)
    return martixList_concat

    #     print(folder_data_read)


# normal_data_read2 = readAsMatrix('044_16280425-G0013/ref_044_16280425-G0013_365364_11080.txt')
# print(np.array(normal_data_read1['JD'][0:50]))
# print(x.shape)
# print(x)
# 将数据切片为n个点构成的小片段
def imgSpilt(data_read,col_name1,col_name2,n):
    x = []
    y = []
    index = 0;
    while (index + n) < (data_read[col_name1].shape[0]):
        # print(data_read[col_name][index:index+50].tolist())
        x.append(data_read[col_name1][index:index + n].tolist())
        y.append(data_read[col_name2][index:index + n].tolist())
        index += n
        # print(index)
    return x,y

# 将文件夹目录的所有数据切成小片段
normalList = readFolderAsMatrix('sourceData/normalData')
targetList = readFolderAsMatrix('sourceData/targetData')
normal_x,normal_y = imgSpilt(normalList,'JD','Magnorm',50)
target_x,target_y = imgSpilt(targetList,'JD','Magnorm',50)


# 将获取一个二维列表，列表中的每一个元素都是50个x,y点对,并将50个点压缩至图片中间
# x1,y1 = imgSpilt(target_dara_read,'JD','Magnorm',50)
# x,y = imgSpilt(normal_data_read1,'JD','Magnorm',50)
# print(y)
# for i in range(len(x)-1):
#     # print(y[i][0])
#     # plt.ylim(y[i][0]-1, y[i][0]+1)
#     # plt.scatter(x[i], y[i], label="normalstar", color="red")
#     plt.ylim(y1[i][0]-1, y1[i][0]+1)
#     plt.scatter(x1[i], y1[i], label="targetstar", color="red")
#     plt.axis('off')


# 批量保存target图片
for i in range(len(target_x)-1):
    img_savePath = 'img/targetImg'
    plt.ylim(target_y[i][0]-2, target_y[i][0]+2)
    plt.scatter(target_x[i], target_y[i], label="targetstar", color="black",s=50)
    plt.axis('off')
    # print(np.var(target_x[i]))
    if not os.path.exists(img_savePath):
        os.makedirs(img_savePath)
    # plt.figure(figsize=(5,5))
    # 把两条不相连的线剔除，并把波动较小的线剔除
    if (np.var(target_y[i]) > 0.001 and np.var(target_x[i]) < 0.01):
        plt.savefig(os.path.join(img_savePath,'{num}'.format(num=i)),dpi = 10,bbox_inches='tight')
    plt.cla()


# 批量保存normal图片
for i in range(len(normal_x)-1):
    img_savePath = 'img/normalImg'
    plt.ylim(normal_y[i][0]-2, normal_y[i][0]+2)
    plt.scatter(normal_x[i], normal_y[i], label="targetstar", color="black",s=50)
    plt.axis('off')
    if not os.path.exists(img_savePath):
        os.makedirs(img_savePath)
    # plt.figure(figsize=(5,5))
    # plt.imshow(plt,cmap='gray')
    # 把不相连的线剔除
    if (np.var(normal_x[i]) < 0.01):
        plt.savefig(os.path.join(img_savePath,'{num}'.format(num=i)),dpi = 10,bbox_inches='tight')
    plt.cla()


# # 将获取两个个二维列表，列表中的每一个元素都是50个x,y点对，将两张图进行对比
# x1,y1 = imgSpilt(target_dara_read,'JD','Magnorm',50)
# x,y = imgSpilt(normal_data_read1,'JD','Magnorm',50)
# for i in range(len(x)):
#     plt.scatter(x[i], y[i], label="normalstar", color="blue")
#     plt.scatter(x1[i], y1[i], label="targetstar", color="red")
#     plt.show()



# targrt_data_read = readAsMatrix('ref_044_16280425-G0013_364820_9174')
# x1 = normal_data_read1['JD'][2000:2050]
# print(x1.shape[0])
# y1 = normal_data_read1['Magnorm'][2000:2050]
# x_target = targrt_data_read['JD'][2000:2050]
# y_target = targrt_data_read['Magnorm'][2000:2050]
# x2 = normal_data_read1['JD'][2000:2050]
# y2 = normal_data_read1['Magnorm'][2000:2050]
# def readnp(filename):
#     f = open(filename)
#     lines = f.readlines()
#     rows = len(lines)
#     for l in lines:
#         col = l.strip('\n').split(' ')
#         print(col)
#     return lines

# x = readnp('044_16280425-G0013/ref_044_16280425-G0013_331771_5921.txt')
# plt.scatter(x1,y1,label = "normalstar",color="blue")
# plt.scatter(x2,y2,label = "normalstar",color="green")
# plt.scatter(x_target,y_target,label = "targetstar",color="red")
#
# plt.legend(loc='upper right')
# plt.show()
