import string
import random
import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt



# 图片路径
IMAGE = r'../img/ztfn.jpg'

# 计算psnr 峰值信噪比


def psnr(imag1, imag2):
    diff = imag1 - imag2
    # print(np.sum(diff))
    mse = np.mean(np.square(diff))
    psnr = 10 * np.log10(255 * 255 / mse)
    return(psnr)

# 计算ssim 结构相似性


def ssim(imag1, imag2):
    (grayScore, diff) = structural_similarity(imag1, imag2, full=True)
    #diff = (diff * 255).astype("uint8")
    return grayScore


def getLastBin(x):
    # 将int x 的最低二进制位 返回
    return int(bin(x)[-1])


def getLSBMatrix(arry):

    lsbMatrix = np.zeros(arry.shape)

    # 获得最低有效位
    for i in range(arry.shape[0]):
        for j in range(arry.shape[1]):
            temp = getLastBin(arry[i][j])
            lsbMatrix[i][j] = temp

    return lsbMatrix


def generateWaterMark(matrix_shape):
    # 随机生成 嵌入字符串 返回  他的lsb矩阵

    string_lsb_matrix = np.zeros(matrix_shape)

    # 生成嵌入信息
    bit_count = matrix_shape[0] * matrix_shape[1]
    Byte_count = bit_count // 8
    # 随机选择一个大小
    rand_string_count = random.randint(1, Byte_count)

    charlist = [random.choice(string.ascii_uppercase)
                for i in range(Byte_count)]
    # 嵌入信息
    chars = ''.join(charlist)

    # 得到嵌入信息的LSB位 保存在一个string里
    stram = ''
    for char in chars:
        char_bin = '0' + bin(ord(char))[2:]
        stram += char_bin

    # 写如lsb到矩阵
    count = 0
    for i in range(matrix_shape[0]):
        for j in range(matrix_shape[1]):
            if count <= len(stram) - 1:
                string_lsb_matrix[i][j] = stram[count]
                count += 1
            else:
                break

    return string_lsb_matrix, chars


def info2matrix(emb_string, original_img):
    '''
    :arg    emb_string:str   需要嵌入到图像的文本信息
            original_img: np.narry  被嵌入图像的矩阵
    :return string_lsb_matrix:文本的LSB矩阵
    将文本信息转换为LSB嵌入矩阵
    '''

    string_lsb_matrix = np.zeros()

    string_len = len(emb_string)
    if string_len > (original_img.shape[0] * original_img.shape[1] // 8):
        print("嵌入信息过长，裁剪")
        emb_string = emb_string[0:(
            original_img.shape[0] * original_img.shape[1] // 8)]

    stram = ''
    for char in chars:
        char_bin = '0' + bin(ord(char))[2:]
        stram += char_bin

    # 写入lsb到矩阵
    count = 0
    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            if count <= len(stram) - 1:
                string_lsb_matrix[i][j] = stram[count]
                count += 1
            else:
                break

    return string_lsb_matrix


def getMatrixDetail(matrix):
    # 得到矩阵的均值和方差并返回
    arry = np.array(matrix)
    mean = np.mean(arry)
    variance = np.mean((arry - mean)**2)
    return mean, variance


def encodeLSB(embedding_matrix, original_matrix):
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

    # 分为 二种情况  m是嵌入比特
    # 1. x = x + m       x = 0 (mod 2) 偶数
    # 2  x = x+m-1       x = 1 (mode 2)  基数数
    # 加入水印
    ans_arry = np.zeros(original_matrix.shape).astype(original_matrix.dtype)
    #最小值从0开始
    for row in range(ori_x):
        for col in range(ori_y):
            if (original_matrix[row][col]) % 2 == 0:
                ans_arry[row][col] = original_matrix[row][col] + embedding_matrix[row][col]
            else:
                ans_arry[row][col] = original_matrix[row][col] + \
                    embedding_matrix[row][col] - 1

    return ans_arry


def show(ori_img, LSB_img,nosie):
    # 图片展示
    plt.rcParams["font.family"] = "SimHei"  # 中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 负号因为中文显示出了错，调整负号显示

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(ori_img, cmap="gray")
    plt.title('原图', fontsize=10)

    plt.subplot(1, 3, 2)
    plt.imshow(LSB_img, cmap="gray")
    plt.title('嵌入LSB信息后', fontsize=10)

    plt.subplot(1, 3, 3)
    plt.imshow(nosie, cmap="gray")
    plt.title('嵌入信息', fontsize=10)



    plt.suptitle('LSB信息嵌入效果对比图', fontsize=20)

    plt.show()


if __name__ == "__main__":
    # 获得灰度图片矩阵
    img_arry = cv.imread(IMAGE, cv.IMREAD_GRAYSCALE)
    # 抽取出最低有效位平面
    original_lsbMatrix = getLSBMatrix(img_arry)

    # 均值 和方差
    mean, variance = getMatrixDetail(original_lsbMatrix)

    embedding_lsbMatrix, chars = generateWaterMark(img_arry.shape)


    encode_LSB_img = encodeLSB(
        embedding_lsbMatrix,
        img_arry)

    print('峰值信噪比（PSNR）为：', psnr(encode_LSB_img, img_arry))
    print('结构相似性（SSIM）为：', ssim(encode_LSB_img, img_arry))
    #
    # cv.imshow("imageLSB", encode_LSB_img)
    # cv.imshow("image", img_arry)
    # cv.waitKey(111110)
    # cv.destroyAllWindows()
    #
    print(chars)
    show(img_arry, encode_LSB_img,embedding_lsbMatrix)
