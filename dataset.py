from typing import Dict, Any, Optional, List
import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations
from albumentations.pytorch import ToTensorV2

PRE_MEAN = [0.485, 0.456, 0.406]
PRE_STD = [0.229, 0.224, 0.225]

def apply_weighted_random_sampler(dataset_csv: str) -> WeightedRandomSampler:
    """
    Crea un sampler ponderado para balancear clases según su frecuencia en el CSV.
    """
    dataframe = pd.read_csv(dataset_csv)  # Se espera columnas: image_path, label
    class_counts = dataframe['label'].value_counts()
    sample_weights = [1.0 / class_counts[label] for label in dataframe['label']]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataframe), replacement=True)
    return sampler


class TrainDataset(Dataset):
    def __init__(self, csv_file: str, input_shape: Optional[tuple] = (224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = albumentations.Compose([
            albumentations.Resize(height=256, width=256),
            albumentations.RandomCrop(height=input_shape[0], width=input_shape[1]),
            albumentations.HorizontalFlip(),
            albumentations.RandomGamma(gamma_limit=(80, 180)),
            albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            albumentations.Normalize(PRE_MEAN, PRE_STD, always_apply=True),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.dataframe)

    def get_labels(self) -> pd.Series:
        return self.dataframe['label']

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unable to load: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = 1 if label_str == 'bonafide' else 0
        map_x = torch.ones((14, 14)) if label == 1 else torch.zeros((14, 14))

        augmented = self.transform(image=image)
        image = augmented['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.float),
            "map": map_x
        }


class TestDataset(Dataset):
    def __init__(self, csv_file: str, input_shape: Optional[tuple] = (224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.Normalize(PRE_MEAN, PRE_STD, always_apply=True),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unable to load: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = 1 if label_str == 'bonafide' else 0

        augmented = self.transform(image=image)
        image = augmented['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.float),
            "img_path": img_path
        }
