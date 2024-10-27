# 基于 SVDD 和手动特征提取的单分类异常检测项目

## 数据准备

选择图像：从 MVTec AD 数据集中选择正常状态的图像（例如瓶盖、电缆、木材或螺母）作为训练集。
同时选择一些正常和有损害的图像作为测试集。

数据预处理：对图像进行预处理，例如调整大小、归一化和数据增强（如旋转、翻转等）来增强模型的鲁棒性。

## 特征提取

选择特征算子：选择适合的特征提取方法，常见的手动特征包括：
边缘检测：如 Canny 边缘检测。
纹理特征：使用灰度共生矩阵（GLCM）计算纹理特征（如对比度、相关性、能量等）。
形状特征：提取形状描述子，如 Hu 矩。
提取特征：对每张图像应用所选的特征算子，将图像转换为特征向量。

## 模型训练

SVDD 模型构建：
使用训练集中的正常状态图像的特征向量来训练 SVDD 模型。
确定超参数（如惩罚参数 C 和核函数），可以通过交叉验证来优化这些参数。
模型训练：训练 SVDD 模型，使其能够将正常样本的特征向量包围在一个超球面内。

## 测试与评估

测试集特征提取：对测试集中的所有图像（正常和有损害的）提取特征。
异常检测：使用训练好的 SVDD 模型对测试集的特征进行分类：
计算每个测试样本到超球面中心的距离，设置一个阈值，超出该阈值的样本被标记为异常。
评估结果：分析模型在测试集上的表现，计算准确率、召回率、F1 分数等指标，并进行可视化展示。



## 要求

1.	此题要求是单分类（One-class classification）,不要以二分类或多分类来实现。即，选择正常状态图片进行模型训练，再选择正常状态图片和有损害图片进行测试，分析检测结果和原因。图片选取数量和类别自行决定，不做统一要求。
2.	采用SVDD或OC-SVM作为基准算法，自行选择合适的图像特征算子，提取手动特征。避免使用深度学习模型和深度特征。
3.	提交内容：代码、实验报告（包括数据集介绍、特征提取方法、模型训练、实验结果分析等）。

数据集下载：https://www.mvtec.com/company/research/datasets/mvtec-ad