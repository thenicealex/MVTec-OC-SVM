from data_module import DATA_PATH, MVTecADDataModule
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage import exposure
from PIL import Image
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class FeatureExtractor:
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def glcm(self, image: Image.Image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(
            gray,
            [1],
            [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels = 256,
            symmetric=True,
            normed=True,
        )
        contrast = graycoprops(glcm, "contrast").flatten()
        dissimilarity = graycoprops(glcm, "dissimilarity").flatten()
        homogeneity = graycoprops(glcm, "homogeneity").flatten()
        energy = graycoprops(glcm, "energy").flatten()
        correlation = graycoprops(glcm, "correlation").flatten()
        return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation]) # (12,)

    def lbp(self, image: Image.Image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # image_cv = cv2.resize(image_cv, (512, 512))
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(image_cv, n_points, radius, method="default")

        n_bins = 8
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 3), range=(0, n_bins+2))
        
        hist = hist.astype("float") / (hist.sum() + 1e-7)
        return hist

    def color_histogram(self, image: Image.Image):
        # image = np.array(image)
        # image_float = image.astype(np.float32)
        # histograms = []
        # n_bins = 256
        # for channel in range(image.shape[2]):
        #     histogram, _ = np.histogram(image_float[:, :, channel], bins=n_bins, range=(0, 256), density=True)
        #     histograms.append(histogram)
        # hist = np.concatenate(histograms)
        hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        bins = (8, 8, 8)
        hist = cv2.calcHist(
            [hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]
        )

        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def hog(self, image: Image.Image):
        image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        image_gray = cv2.resize(image_gray, (512, 512))
        feat = hog(
            image_gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        return feat
    
    def extract(self, image: Image.Image, hog = True, lbp=False, glcm=False, color_histogram=False):
        # Initialize the output with HOG features
        features = np.array([])

        # Define a dictionary to map feature flags to their corresponding methods
        feature_methods = {
            hog: self.hog,
            lbp: self.lbp,
            glcm: self.glcm,
            color_histogram: self.color_histogram
        }

        # Iterate over the dictionary and extract features if the flag is True
        for flag, method in feature_methods.items():
            if flag:
                extracted_feature = method(image)
                features = np.hstack([features, extracted_feature])

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
    f = ex.extract(image, hog=False, color_histogram=True)
    print(f.shape)
