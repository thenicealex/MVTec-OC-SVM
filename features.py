import torch
from data_module import DATA_PATH, MVTecADDataModule
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage import exposure
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import tensor_to_PIL
from skimage.feature import graycomatrix, graycoprops


class FeatureExtractor:
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def ORB(self, image: Image.Image):
        open_cv_image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        
        orb = cv2.ORB_create()
        
        keypoints, descriptors = orb.detectAndCompute(open_cv_image, None)
        
        # 如果没有检测到特征点，返回一个空数组
        if descriptors is None:
            return np.array([])
        
        return descriptors.flatten()

    def sobel(self, image: Image.Image):
        # 读取图像
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # 应用Sobel算子
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        pool_size = (4, 4)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        pooled_features = cv2.resize(
            sobel_magnitude,
            (
                sobel_magnitude.shape[1] // pool_size[1],
                sobel_magnitude.shape[0] // pool_size[0],
            ),
            interpolation=cv2.INTER_AREA,
        )

        return pooled_features.flatten()

    def glcm(self, image: Image.Image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(
            gray,
            [1],
            [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            symmetric=True,
            normed=True,
        )
        contrast = graycoprops(glcm, "contrast").flatten()
        dissimilarity = graycoprops(glcm, "dissimilarity").flatten()
        homogeneity = graycoprops(glcm, "homogeneity").flatten()
        energy = graycoprops(glcm, "energy").flatten()
        return np.hstack([contrast, dissimilarity, homogeneity, energy])

    def sift(self, image: Image.Image):
        # 特征点检测和描述
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # image_cv is a numpy array with shape (height, width, 3)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image_cv, None)
        return keypoints, descriptors

    def canny(self, image: Image.Image):
        # 边缘特征提取
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        edges = cv2.Canny(image_cv, threshold1=100, threshold2=200)
        return edges

    def lbp(self, image: Image.Image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(image_cv, n_points, radius, method="uniform")
        return lbp

    def color_histogram(self, image: Image.Image):
        hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        bins=(8, 8, 8)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def hog(self, image: Image.Image):
        image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        image_gray = cv2.resize(image_gray, (256, 256))
        feat = hog(
            image_gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        return feat

    def extract(self, image: Image.Image):
        # Example of extracting all features
        features = {
            "sift": self.sift(image)[1],
            "canny": self.canny(image),
            "lbp": self.lbp(image),
            "color_histogram": self.color_histogram(image),
            "hog": self.hog(image)[0],
        }
        return features


def process_and_save_features(dataset, image_index=0):
    extractor = FeatureExtractor(dataset)
    # Draw image
    image = dataset[image_index]
    # image.save('image.png')

    # Draw hog feature map
    fd, hog_image = extractor.hog(image)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_image_pil = Image.fromarray((hog_image * 255).astype(np.uint8))
    # hog_image_pil.save('hog.png')

    # Draw lbp feature map
    lbp = extractor.lbp(image)
    lbp_image_pil = Image.fromarray((lbp * 255).astype(np.uint8))
    # lbp_image_pil.save('lbp.png')

    # Draw surf feature map
    keypoints, surf = extractor.canny(image)
    image_with_keypoints = cv2.drawKeypoints(
        np.array(image),
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    canny_image_pil = Image.fromarray(
        cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
    )
    # canny_image_pil.save('canny.png')

    # Draw sift feature map
    keypoints, sift = extractor.sift(image)
    image_with_keypoints = cv2.drawKeypoints(
        np.array(image),
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    sift_image_pil = Image.fromarray(
        cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
    )
    # sift_image_pil.save('sift.png')

    # Concatenate all feature images into one long image
    # images = [Image.open(f) for f in ['image.png', 'hog.png', 'lbp.png', 'surf.png', 'sift.png']]
    images = [image, hog_image_pil, lbp_image_pil, canny_image_pil, sift_image_pil]

    for i in range(len(images)):
        width, height = images[i].size
        print(f"Image {i+1}: Width = {width}, Height = {height}")

    height = images[0].height
    total_width = sum(image.width for image in images)

    concatenated_image = Image.new("RGB", (total_width, height))

    x_offset = 0
    for image in images:
        concatenated_image.paste(image, (x_offset, 0))
        x_offset += image.width

    concatenated_image.save("features.png")


if __name__ == "__main__":
    train_dataset = MVTecADDataModule(DATA_PATH, mode="train", category="bottle")
    test_dataset = MVTecADDataModule(DATA_PATH, mode="test", category="bottle")
    # process_and_save_features(dataset, image_index=0)
    ex = FeatureExtractor()
    image, _ = train_dataset[0]
    # image = tensor_to_PIL(image)
    print(image)
    f = ex.hog(image)
    print(f.shape)
    print(f)
