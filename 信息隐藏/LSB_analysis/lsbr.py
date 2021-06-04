from collections import Counter
import itertools
# 添加路径
import sys
sys.path.append('/home/zentreisender/Documents/imformation/信息隐藏')
from LSB.lsbSub import *


# 统计
def odd_even(ima):
    # 行数
    if ima.shape[0] % 2 != 0:
        print("error, 行数不是偶数")
        return

    # 列数
    if ima.shape[1] % 2 != 0:
        print("error, 列数不是偶数")
        return

    count_odd = 0
    count_even = 0
    for i in range(ima.shape[0]):
        for j in range(0, ima.shape[1], 2):
            # 数据以uint8存储，偶数0，2，4 奇数1，3，5
            # 如果是奇偶
            if ima[i][j] % 2 == 1 & ima[i][j+1] % 2 == 0:
                # 如果奇数大于偶数
                if ima[i][j] > ima[i][j+1]:
                    count_odd += 1
                elif ima[i][j] < ima[i][j+1]:
                    count_even += 1
            # 如果是偶奇
            elif ima[i][j] % 2 == 0 & ima[i][j+1] % 2 == 1:
                # 如果偶数小于奇数
                if ima[i][j] < ima[i][j+1]:
                    count_odd += 1
                elif ima[i][j] > ima[i][j+1]:
                    count_even += 1

    return count_odd, count_even


if __name__ == "__main__":
    # 图片路径
    IMAGE = r'../img/ztfn.jpg'
    # 获得灰度图片矩阵
    img_arry = cv.imread(IMAGE, cv.IMREAD_GRAYSCALE)
    # 抽取出最低有效位平面
    original_lsbMatrix = getLSBMatrix(img_arry)

    # 均值 和方差
    #mean, variance = getMatrixDetail(original_lsbMatrix)
    count_oddo, count_eneno = odd_even(img_arry)
    p = []
    p.append(count_oddo/count_eneno)
    img = []
    img.append("origin")
    for i in range(6):

        embedding_lsbMatrix, chars = generateWaterMark(img_arry.shape)

    #print("嵌入量：%d" %(embedding_lsbMatrix.size))
    # print("嵌入值："+chars)
        encode_LSB_img = encodeLSB(
            embedding_lsbMatrix,
            img_arry)

        count_odde, count_enene = odd_even(encode_LSB_img)
        p.append(count_odde/count_enene)
        img.append("LSBR"+str(i))

    x = range(len(img))
    rects1 = plt.bar(x=x, height=p, width=0.4, alpha=0.8,color='black',label="ratio=(odd>enen) / (even>odd)")
    plt.ylim(0, 1.2)     # y轴取值范围
    plt.ylabel("ratio")

    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """

    plt.xticks([index for index in x], img)
    plt.xlabel("img")
    plt.title("")
    plt.legend()     # 设置题注

    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")
 
    plt.show()