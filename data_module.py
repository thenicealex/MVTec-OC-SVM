import os
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from typing import Union, List
from PIL import Image
from utils import tensor_to_PIL

DATA_PATH = "/home/pod/shared-nvme/datasets/MVTec-AD"


class MVTecADTrainDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        category: str = "bottle",
    ):
        self.root_dir = root_dir
        self.mode = "train"

        self.category = category
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        return os.path.join(self.root_dir, self.category, self.mode, "good")

    def __len__(self):
        return len(os.listdir(self.image_paths))

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_paths, os.listdir(self.image_paths)[idx])
        labels = [1] * len(os.listdir(self.image_paths))

        image = Image.open(image_path).convert("RGB")
        label = np.array(labels[idx])
        return image, label


class MVTecADTestDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        category: str = "bottle",
    ):
        self.root_dir = root_dir
        self.mode = "test"

        self.category = category
        self.image_paths = self._get_image_paths()
        self.good_image_paths = os.path.join(self.image_paths, "good")
        self.anomalous_image_paths = self._get_anomalous_image_paths()

    def _get_anomalous_image_paths(self):
        anomalous_categories = [
            path for path in os.listdir(self.image_paths) if path != "good"
        ]
        return [os.path.join(self.image_paths, c) for c in anomalous_categories]

    def _get_image_paths(self):
        return os.path.join(self.root_dir, self.category, self.mode)

    def __len__(self):
        return len(os.listdir(self.good_image_paths)) + sum(
            len(os.listdir(p)) for p in self.anomalous_image_paths
        )

    def _gather_images_and_labels(self):
        images_path = [
            os.path.join(self.good_image_paths, p)
            for p in os.listdir(self.good_image_paths)
        ]
        labels = [1] * len(images_path)
        for ap in self.anomalous_image_paths:
            anomalous_images = [os.path.join(ap, p) for p in os.listdir(ap)]
            images_path.extend(anomalous_images)
            labels.extend([-1] * len(anomalous_images))
        return images_path, labels

    def __getitem__(self, idx):
        images_path, labels = self._gather_images_and_labels()
        image = Image.open(images_path[idx]).convert("RGB")
        label = np.array(labels[idx])
        return image, label

class MVTecADTestAnomalousDataset(Dataset):
    def __init__(self, root_dir: str, category: str = "bottle", anomalous_category: str = "broken_small"):
        super().__init__()
        self.root_dir = root_dir

        self.category = category
        self.image_paths = self._get_image_paths()

        self.anomalous_image_paths = os.path.join(self.image_paths, anomalous_category)
        
    def _get_image_paths(self):
        return os.path.join(self.root_dir, self.category, "test")

    def __len__(self):
        return len(os.listdir(self.anomalous_image_paths))
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.anomalous_image_paths, os.listdir(self.anomalous_image_paths)[idx])
        image = Image.open(image_path).convert("RGB")
        label = np.array([-1])
        return image, label

class MVTecADDataModule(Dataset):
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

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        category: str = "bottle",
    ):
        self.root_dir = root_dir
        self.mode = mode

        assert category in self.CLASSES
        self.category = category
        self.dataset_generator = self._get_dataset_generator()

    def _get_dataset_generator(self):
        if self.mode == "train":
            return MVTecADTrainDataset(self.root_dir, category=self.category)
        return MVTecADTestDataset(self.root_dir, category=self.category)

    def __len__(self):
        return len(self.dataset_generator)

    def __getitem__(self, idx):
        image, label = self.dataset_generator[idx]
        return image, label


if __name__ == "__main__":
    # dataset = MVTecADDataModule(DATA_PATH, mode="train", category="bottle")
    # print(len(dataset))
    # image, _ = dataset[5]
    # print(image)
    # image.save("image.png")

    da = MVTecADTestAnomalousDataset(DATA_PATH, category="bottle", anomalous_category="broken_small")
    print(len(da))
    image, _ = da[0]
    image.save("image.png")
