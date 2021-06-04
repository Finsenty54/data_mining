from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import itertools
import sys
# 添加路径
sys.path.append('/home/zentreisender/Documents/imformation/信息隐藏')
from LSB.lsbSub import *
from LSB.lsbm import *


# 计算邻域度，k=2，表示5*5矩阵
def nei(ima):
    imaMAT=np.mat(ima)

    # 记录像素点邻域度
    d = np.zeros(imaMAT.size, dtype=int)
    # 遍历有完整邻域的像素点
    for i in range(2,imaMAT.shape[0]-2):
        for j in range(2,imaMAT.shape[1]-2):
            # 遍历每个相邻域内的点
            for u in range(2, -3, -1):
                for v in range(2, -3, -1):
                    # 不和自身比较
                    if not(u == 0 and v == 0):
                        # 相等则记录
                        if imaMAT[i+u,j+u] == imaMAT[i,j]:
                            d[i*imaMAT.shape[0]+j] += 1

    return d


if __name__ == "__main__":
    for i in range(4):
        # 图片路径
        IMAGE = r'../img/'+str(i+1)+'.tiff'
        # 获得灰度图片矩阵
        img_arry = cv.imread(IMAGE, cv.IMREAD_GRAYSCALE)

        embedding_lsbMatrix, chars = generateWaterMark(img_arry.shape)

        encode_LSBm = encodeLSBM(embedding_lsbMatrix, img_arry)

        print("1")
        # print(encode_LSB_img)
        d_encode = nei(encode_LSBm)
        d_unencode = nei(img_arry)

        # 图片展示
        #fig, ax = plt.subplots(1, 2, figsize=(10, 6))

        ax1=plt.subplot(2, 4, i*2+1)
        plt.hist(d_unencode, range(27))
        plt.title('original', fontsize=10)
        ax2=plt.subplot(2, 4, i*2+2,sharey=ax1)
        plt.hist(d_encode, range(27))
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.title('LSBM'+str(i+1), fontsize=10)

    plt.suptitle('LSBM_analysis', fontsize=20)

    plt.show()

    # print(chars)
    #show(img_arry, encode_LSB_img,embedding_lsbMatrix)
