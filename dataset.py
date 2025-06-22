import pandas as pd
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, csv_file: str, feat_file: str):
        self.dataframe = pd.read_csv(csv_file)
        self.feat_dict = {}
        with open(feat_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                path = parts[0]
                feats = list(map(float, parts[1:]))
                self.feat_dict[path] = torch.tensor(feats, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        feature = self.feat_dict.get(img_path)
        if feature is None:
            raise ValueError(f"Feature not found for image path: {img_path}")

        label = 1 if label_str == 'bonafide' else 0

        return feature, torch.tensor(label, dtype=torch.long), img_path

class TestDataset(TrainDataset):
    pass
