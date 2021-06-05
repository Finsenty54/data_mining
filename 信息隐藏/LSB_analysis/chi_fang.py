from collections import Counter
import itertools
import sys
# 添加路径
sys.path.append('/home/zentreisender/Documents/data_mining/信息隐藏')
from LSB.jsteg import *
from LSB.lsbSub import *


# 返回卡方差总和
def chi_fang(ima):

    # 将矩阵转为数组形式
    List = list(itertools.chain.from_iterable(ima))
    arr = np.array(List)

    print(min(arr))
    print(max(arr))

    # 获得像素值对应的个数
    result = {}
    for k in range(256):
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v

    # 计算相临值的卡方差
    sum = 0
    for i in range(0, 128):
        # 计算卡方差总和
        sum = pow(result[2*i]-result[2*i+1], 2)+sum

    return sum, result


if __name__ == "__main__":
    # 图片路径
    IMAGE = r'../img/lena256.bmp'
    # 获得灰度图片矩阵
    img_arry = cv.imread(IMAGE, cv.IMREAD_GRAYSCALE)

    print(img_arry.dtype.name)

    embedding_lsbMatrix1, chars1 = generateWaterMark(img_arry.shape)
    embedding_lsbMatrix2, chars2 = generateWaterMark(img_arry.shape)

    # 嵌入随机生成的数据
    # embedding_lsbMatrix1=np.random.randint(0,2,img_arry.shape)

    encode_LSB_img1 = encodeLSB(
        embedding_lsbMatrix1,
        img_arry)
    encode_LSB_img2 = encodeLSB(
        embedding_lsbMatrix2,
        img_arry)


    # print(encode_LSB_img)
    chi_encode1, result1 = chi_fang(encode_LSB_img1)
    chi_encode2, result2 = chi_fang(encode_LSB_img2)
    chi_unencode, result = chi_fang(img_arry)

    # 打印原图像卡方差，嵌入后图像卡方差
    print("chi_unencode: %d , chi_encode: %d" % (chi_unencode, chi_encode1))
    plt.figure()

    plt.subplot(2, 3, 1)
    plt.imshow(img_arry, cmap="gray")
    plt.title('origin', fontsize=10)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(encode_LSB_img1, cmap="gray")
    plt.title('LSBR1', fontsize=10)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(embedding_lsbMatrix1, cmap="gray")
    plt.title('noise1', fontsize=10)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(img_arry, cmap="gray")
    plt.title('origin', fontsize=10)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(encode_LSB_img2, cmap="gray")
    plt.title('LSBR2', fontsize=10)

    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(embedding_lsbMatrix2, cmap="gray")
    plt.title('noise2', fontsize=10)

    plt.axis('off')
    #plt.suptitle('LSB信息嵌入效果对比图', fontsize=20)

    plt.show()

    plt.figure()
    x = ["origin", "LSBR1", "LSBR2"]
    y = [chi_unencode, chi_encode1, chi_encode2]
    plt.plot(x, y)

    plt.figure()
    x = np.arange(0, 256)
    l1 = plt.plot(list(result.keys()), list(
        result.values()), 'r--', label='origin')
    l2 = plt.plot(list(result1.keys()), list(
        result1.values()), 'g--', label='LSBR1')
    l3 = plt.plot(list(result2.keys()), list(
        result2.values()), 'b--', label='LSBR2')
    plt.plot(list(result.keys()), list(result.values()),
             'ro-', list(result1.keys()), list(result1.values()),
             'g+-', list(result2.keys()), list(result2.values()), 'b^-')
    plt.title('Pixel distribution')
    plt.xlabel('pixel')
    plt.ylabel('count')
    plt.legend()

    plt.show()
