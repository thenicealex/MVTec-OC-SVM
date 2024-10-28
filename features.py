from data_module import DATA_PATH, MVTecADDataModule
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage import exposure
from PIL import Image
import numpy as np
from skimage.feature import graycomatrix, graycoprops


class FeatureExtractor:
    def __init__(self):
        super(FeatureExtractor, self).__init__()

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
        return np.hstack([contrast, dissimilarity, homogeneity, energy]) # (12,)

    def canny(self, image: Image.Image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(image_cv, threshold1=100, threshold2=200)
        return edges.flatten() #(810000,)

    def lbp(self, image: Image.Image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(image_cv, n_points, radius, method="uniform")

        grid_x, grid_y = 30, 30
        features = []

        for i in range(0, image_cv.shape[0], grid_y):
            for j in range(0, image_cv.shape[1], grid_x):
                block = lbp[i : i + grid_y, j : j + grid_x]

                hist, _ = np.histogram(
                    block.ravel(), bins=np.arange(0, 26), range=(0, 26)
                )

                hist = hist.astype("float")
                hist /= hist.sum() + 1e-6

                features.append(hist)

        features = np.concatenate(features, axis=0)
        return features

    def color_histogram(self, image: Image.Image):
        hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        bins = (8, 8, 8)
        hist = cv2.calcHist(
            [hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]
        )

        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def hog(self, image: Image.Image):
        image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        # image_gray = cv2.resize(image_gray, (256, 256))
        feat = hog(
            image_gray,
            orientations=9,
            pixels_per_cell=(32, 32),
            cells_per_block=(3, 3),
            feature_vector=True,
        )
        return feat
    
    def extract(self, image: Image.Image):
        hog = self.hog(image)
        # print("hog shape: ", hog.shape) # (34596,)

        color_h = self.color_histogram(image)
        # print("color histogram shape: ", color_h.shape) # (512,)

        lbp = self.lbp(image)
        # print("lbp shape: ", lbp.shape) # (22500,)
        
        glcm = self.glcm(image)
        
        canny = self.canny(image)
        
        # return np.hstack([lbp, glcm, hog])
        return hog


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
    print(image)
    f = ex.ORB(image)
    # print(f)
    print(f.shape)
