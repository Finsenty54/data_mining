import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

#  量化矩阵
FN_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


class Jsteg:
    def __init__(self, target_matrix, embedd_info: list, N):
        """:arg embedd_info:list  bit流"""
        self.target_matrix = target_matrix
        self.embedd_info = embedd_info
        self.N = N
        self.length = len(self.embedd_info)

    def orderEncodeAcLSB(self):
        ''':arg
        对源图像进行AC4隐写
        输出隐写玩后的图像
        '''
        height, width = self.target_matrix.shape
        # 分为 二种情况  m是嵌入比特
        # 1. x = x + m       x = 0 (mod 2) 偶数
        # 2  x = x+m-1       x = 1 (mode 2)  基数数
        # 加入水印
        ans_arry = self.target_matrix.copy()
        count = 0  # 计数器
        for row in np.arange(0, height, self.N):
            for col in np.arange(0, width, self.N):
                # 超出 跳出嵌入
                if count > self.length - 1:
                    break
                # 取块
                block = np.array(
                    img[row:(row + self.N), col:(col + self.N)], dtype=np.int32)

                # 题目中对第4个AC系数进行嵌入
                if block[1][1] % 2 == 0:
                    ans_arry[row + 1][col + 1] = self.target_matrix[row + \
                        1][col + 1] + self.embedd_info[count]

                else:
                    ans_arry[row + 1][col + 1] = self.target_matrix[row + \
                        1][col + 1] + self.embedd_info[count] - 1

                count += 1

        return ans_arry

    def randomEncodeAcLSB(self):
        """
        随机嵌入
        """
        p_list = 0  # 列表指针
        block_count = 0  # 计数指针

        # 随机选取128个bit
        select_range = range(32 * 32)
        select = sorted(random.sample(select_range, 128))

        height, width = self.target_matrix.shape
        # 分为 二种情况  m是嵌入比特
        # 1. x = x + m       x = 0 (mod 2) 偶数
        # 2  x = x+m-1       x = 1 (mode 2)  基数数
        # 加入水印
        ans_arry = self.target_matrix.copy()
        count = 0  # 计数器
        for row in np.arange(0, height - self.N, self.N):
            for col in np.arange(0, width - self.N, self.N):
                # 超出 跳出嵌入
                if count > self.length:
                    break

                # 这个块不是 需要嵌入的块
                if block_count != select[p_list]:
                    block_count += 1
                    continue

                # 取块
                block = np.array(
                    img[row:(row + self.N), col:(col + self.N)], dtype=np.int32)
                # 题目中对第4个AC系数进行嵌入

                if block[1][1] % 2 == 0:
                    ans_arry[row + 1][col + 1] = self.target_matrix[row + \
                        1][col + 1] + self.embedd_info[count]
                else:
                    ans_arry[row + 1][col + 1] = self.target_matrix[row + \
                        1][col + 1] + self.embedd_info[count] - 1
                count += 1
                block_count += 1
                p_list += 1
        return ans_arry


def dctEncode(data_matrix):
    """
    计算dct编码 返回dct编码系数
    """
    m, n = data_matrix.shape

    # 变成对称的
    data_matrix = data_matrix - 128 * np.ones([m, n])
    temp = np.zeros(data_matrix.shape)
    dct = np.zeros(data_matrix.shape)

    N = n
    temp[0, :] = 1 * np.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            temp[i][j] = np.cos(np.pi * i * (2 * j + 1) /
                                (2 * N)) * np.sqrt(2 / N)

    dct = np.dot(temp, data_matrix)
    dct = np.dot(dct, np.transpose(temp))

    return np.round(dct, 2)  # 保留两位小数


def decodeDCT(dct):
    """
    将dct后的矩阵 变回dct的矩阵
    :param dct:
    :return:
    """
    m, n = dct.shape

    # 变成对称的
    # data_matrix = data_matrix - 128 * np.ones([m, n])
    temp = np.zeros(dct.shape)

    N = n
    temp[0, :] = 1 * np.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            temp[i][j] = np.cos(np.pi * i * (2 * j + 1) /
                                (2 * N)) * np.sqrt(2 / N)

    img_recor = np.dot(np.transpose(temp), dct)
    img_recor1 = np.dot(img_recor, temp)

    return np.round(img_recor1)


def generate_bit(n):
    # 生成水印比特list
    embedd_info_list = []
    count = 0
    while count < n:
        embedd_info_list.append(random.choice((0, 1)))
        count += 1

    return embedd_info_list


def block_quantization(img, N):
    # N 是格子块
    # 返回量化系数矩阵
    height, width = img.shape

    dct_matrix = np.zeros((height, width), dtype=np.int32)

    for row in np.arange(0, height, N):
        for col in np.arange(0, width, N):
            # 取块
            block = np.array(
                img[row:(row + N), col:(col + N)], dtype=np.float32)
            dct_matrix[row:(row + N), col:(col + N)
                       ] = np.round(dctEncode(block) / FN_MATRIX)

    return dct_matrix


def anti_block_quantization(dct_matrix, N):
    # N 是格子块
    # 返回原始图像
    height, width = dct_matrix.shape

    anti_dct_matrix = np.zeros((height, width), dtype=np.float32)

    for row in np.arange(0, height, N):
        for col in np.arange(0, width, N):
            # 取块
            block = np.array(
                dct_matrix[row:(row + N), col:(col + N)], dtype=np.float32)

            anti_dct_matrix[row:(row + N), col:(col + N)
                            ] = decodeDCT(block * FN_MATRIX) + 128 * np.ones([8, 8])

    return anti_dct_matrix


def draw(img, dct_quantization_matrix, res_lsb, lsb_recover_image):
    plt.subplot(221)
    plt.imshow(img, 'gray')
    plt.title('original image')

    plt.subplot(222)
    plt.imshow(dct_quantization_matrix, 'gray')
    plt.title('dct')

    plt.subplot(223)
    plt.imshow(res_lsb, 'gray')
    plt.title('lsb dct')

    plt.subplot(224)
    plt.imshow(lsb_recover_image, 'gray')
    plt.title('orderLsbEmbedding image')

    plt.title('jsteg Embedding')
    plt.show()


def getAClist(ori_img, lsb_img, N):

    height, width = ori_img.shape
    oriImgACList = []
    lsbImgACList = []

    for row in np.arange(0, height, N):
        for col in np.arange(0, width, N):
            ori_block = np.array(
                ori_img[row:(row + N), col:(col + N)], dtype=np.int32)

            lsb_block = np.array(
                lsb_img[row:(row + N), col:(col + N)], dtype=np.int32)

            oriImgACList.append(ori_block[1][1])
            lsbImgACList.append(lsb_block[1][1])

    return oriImgACList, lsbImgACList


def drawHist(oriImgACList, lsbImgACList):

    plt.subplot(121)
    plt.hist(oriImgACList, bins=6, range=(-6, 6))
    plt.grid(True)
    plt.xticks([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    plt.title('originals image')

    plt.subplot(122)
    plt.hist(lsbImgACList, bins=6, range=(-6, 6))
    plt.xticks([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    plt.grid(True)
    plt.title('lsb image')
    plt.show()


if __name__ == '__main__':
    # 分成8*8的矩阵
    N = 8
    info_bit_count = 128  # 嵌入128bit信息

    IMAGE = r'../img/lena.jpg'

    img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
    # 得到量化后的系数矩阵
    dct_quantization_matrix = block_quantization(img, N)
    # 需嵌入信息
    embedd_info_list = generate_bit(info_bit_count)
    # jsteg 算法
    js = Jsteg(dct_quantization_matrix, embedd_info_list, N)
    # 顺序嵌入
    res_lsb = js.orderEncodeAcLSB()
    # # 恢复图像
    # lsb_recover_image = anti_block_quantization(res_lsb, N)
    # draw(img, dct_quantization_matrix, res_lsb, lsb_recover_image)
    # # 获取AC系数
    # OrderOriImgACList, OrderLsbImgACList = getAClist(
    #     dct_quantization_matrix, res_lsb, N)
    # print(max(OrderOriImgACList), min(OrderOriImgACList))
    # drawHist(OrderOriImgACList, OrderLsbImgACList)

    random_lsb = js.randomEncodeAcLSB()
    # 恢复图像
    lsb_recover_image = anti_block_quantization(random_lsb, N)
    draw(img, dct_quantization_matrix, random_lsb, lsb_recover_image)
    # 获取AC系数
    OrderOriImgACList, OrderLsbImgACList = getAClist(
        dct_quantization_matrix, random_lsb, N)
    print(max(OrderOriImgACList), min(OrderOriImgACList))
    drawHist(OrderOriImgACList, OrderLsbImgACList)
