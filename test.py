import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
import cv2
import numpy as np

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    提取图像的颜色直方图特征并将其展平为特征向量。

    参数:
    - image: 输入图像（BGR格式）。
    - bins: 每个颜色通道的分箱数（默认为8x8x8）。

    返回:
    - histogram: 展平的颜色直方图特征向量。
    """
    # 将图像从BGR格式转换为HSV格式
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 计算HSV三通道的直方图
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    
    # 归一化直方图，使其不受图像尺寸的影响
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

# 读取示例图像
pic_file = '/home/pod/shared-nvme/datasets/MVTec-AD/bottle/train/good/000.png'
image = cv2.imread(pic_file)
histogram_features = extract_color_histogram(image)

print("颜色直方图特征向量的长度:", len(histogram_features))
print("颜色直方图特征向量:", histogram_features.shape)