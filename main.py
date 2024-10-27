import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from data_module import DATA_PATH, MVTecADDataModule
from utils import check_shape, tensor_to_PIL
from features import FeatureExtractor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
import logging
import matplotlib.pyplot as plt

CLASSES = [
    "bottle",
    "carpet",
    "leather",
    "pill",
    "tile",
    "wood",
    "cable",
    "grid",
    "toothbrush",
    "zipper",
    "capsule",
    "hazelnut",
    "metal_nut",
    "screw",
    "transistor",
]

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_dataset(mode, category="bottle"):
    logging.info(f"Loading {mode} dataset...")
    return MVTecADDataModule(DATA_PATH, mode=mode, category=category)


def train_model(features):
    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.1)
    model.fit(features)
    logging.info("Model training completed.")
    return model


def evaluate_model(model, test_features, test_labels, category="bottle"):
    logging.info(f"Test features num: {len(test_features)}")
    predictions = model.predict(test_features)
    normal_count = sum(predictions == 1)
    anomaly_count = sum(predictions == -1)
    logging.info(f"Normal images: {normal_count}, Anomalous images: {anomaly_count}")

    auc = roc_auc_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)

    logging.info(
        f"For {category}, AUC: {auc}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}"
    )

    # return predictions, auc, f1, precision, recall


def extract_features(ex, train_dataset, test_dataset):

    def extract(dataset):
        features = [ex.hog(image) for image, _ in dataset]
        labels = [data[1].item() for data in dataset]
        return features, labels

    train_features, train_labels = extract(train_dataset)
    test_features, test_labels = extract(test_dataset)

    return train_features, train_labels, test_features, test_labels


def train_and_evaluate_all_category():
    for category in CLASSES:
        logging.info(f"Processing category: {category}")
        train_dataset = load_dataset(mode="train", category=category)
        test_dataset = load_dataset(mode="test", category=category)

        logging.info("Extracting features...")
        ex = FeatureExtractor()
        train_features, train_labels, test_features, test_labels = extract_features(
            ex, train_dataset, test_dataset
        )

        logging.info("Training the OC-SVM model...")
        model = train_model(train_features)

        logging.info("Evaluating the model...")
        evaluate_model(model, test_features, test_labels, category=category)


def train_and_evaluate_one_category(category: str = "bottle"):
    logging.info(f"Processing category: {category}")
    train_dataset = load_dataset(mode="train", category=category)
    test_dataset = load_dataset(mode="test", category=category)

    logging.info("Extracting features...")
    ex = FeatureExtractor()
    train_features, train_labels, test_features, test_labels = extract_features(
        ex, train_dataset, test_dataset
    )

    logging.info("Training the OC-SVM model...")
    model = train_model(train_features)

    logging.info("Evaluating the model...")
    evaluate_model(model, test_features, test_labels, category=category)


def main():
    # train_and_evaluate_one_category("wood")
    train_and_evaluate_all_category()


if __name__ == "__main__":
    main()

    # pca = PCA(n_components=0.95)  # 保留95%的方差
    # train_features = pca.fit_transform(train_features)
    # test_features = pca.transform(test_features)

    # scaler = StandardScaler()
    # train_features = scaler.fit_transform(train_features)
    # test_dataset = scaler.transform(test_features)
