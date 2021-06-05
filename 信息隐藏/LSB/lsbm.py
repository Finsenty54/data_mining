# --coding :utf-8 --
# @time:    2021/3/15 8:45 上午
# @IDE:     pythonProject1
# @Author:  ZhangTuo
# @Email:   1271899330@qq.com
# @File:    lsbm_demo.py

from PIL import Image
import random
import numpy as np
import sys
#添加路径
sys.path.append('/home/zentreisender/Documents/imformation/信息隐藏')
import LSB.lsbSub

'''
获取图像像素点，并转化为二进制
image:原始图片
changepixs：[['1','0'],['1','1']]形式列表

'''
def getpix(image):
    pix = list(image.getdata())   #获取像素点，int形列表
    changepixs = []

    for cou in pix:                     #取列表中每一个像素点
        newpix = list('0'*(8-(len(bin(cou))-2))+bin(cou).replace('0b',''))   #转化为二进制bit，0B去除，并补齐000，
        changepixs.append(newpix)               #    changepixs 是字符列表构成的列表
    # print(changepixs)
    return changepixs

'''
转换为二进制，供调用
'''
def consLenBin(int):
    binary = '0'*(8-(len(bin(int))-2))+bin(int).replace('0b','')
    return binary



'''
获取隐藏数据，并转化为二进制
data：待隐藏数据
binary：待隐藏数据二进制流
'''
def getdatabitarry(data):
    binary = ''.join(map(consLenBin,bytearray(data,'utf-8')))   #join拼接二进制流
    return binary


'''
将二进制隐藏信息编码进图片
image： 原始图片
data：待隐藏数据
'''
def encodeImage(image,data):

    imagepixs = getpix(image)            #调用函数获得图片二进制列表
    databittary = getdatabitarry(data)   #调用函数获取数据二进制流

    for i in range(len(databittary)):          #取图片前 比特流长度的  像素点
        if imagepixs[i][-1] != databittary[i]:              #若比特与LSB不同，则随机加减一

            num = int(''.join('%s' %it for it in imagepixs[i]),2)   #先把二进制列表转换成二进制字符串，再转成十进制
            if num==255 or num==0:        #若该像素点是 255 或0 则跳过，防止溢出
                continue
            num = num + random.choice((1,-1))    #随机加减一
            imagepixs[i] = list('0'*(8-(len(bin(num))-2))+bin(num).replace('0b',''))   #重新转成二进制

    resultpixs = []
    for item in imagepixs:
        new_str = ''.join('%s' %it for it in item)     #去  '，'  ，转化为比特流字符串
        resultpixs.append(int(new_str,2))    #修改好的数据解码成int形列表

    encodeimage = Image.new(image.mode,image.size)    #获取编码后图像的模型尺寸
    encodeimage.putdata(resultpixs)    #编码进图像
    return encodeimage

def encodeLSBM(embedding_matrix, original_matrix):
    # 将嵌入LSB矩阵 embedding_matrix 嵌入到 原图矩阵original_matrix中去
    ori_x = original_matrix.shape[0]
    embe_x = embedding_matrix.shape[0]
    ori_y = original_matrix.shape[1]
    embe_y = embedding_matrix.shape[1]
    # 考虑边界问题 大了就裁剪
    if(ori_x < embe_x or ori_y < embe_y):
        print('嵌入信息太大，将裁剪信息！/n')
        embedding_matrix = embedding_matrix[0:ori_x, 0:ori_y]
        print("裁剪后的信息大小为{}".format(embedding_matrix.shape))

    # 分为 四种种情况  m是嵌入比特
    # 1. x = x       x = 0 (mod 2)   0
    # 2  x = x       x = 1 (mode 2)  1
    # 3  x = x+-1    x = 0 (mode 2)  1
    # 4  x = x+-1    x = 1 (mode 2)  0
    # 加入水印
    ans_arry = np.zeros(original_matrix.shape).astype(original_matrix.dtype)

    for row in range(embe_x):
        for col in range(embe_y):
            if (original_matrix[row][col] % 2) != embedding_matrix[row][col]:
                ans_arry[row][col] = (original_matrix[row][col] + random.choice((1.,-1.)))% 256

    return ans_arry

if __name__ == '__main__':
    encodeImage(Image.open('lena256.bmp'),'你好世界').save('demo.bmp')


