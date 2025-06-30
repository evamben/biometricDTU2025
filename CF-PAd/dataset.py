import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class CasiaFasdDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform

        for root, _, files in os.walk(img_dir):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                if 'real' in fname.lower():
                    label = 1
                elif 'fake' in fname.lower() or 'spoof' in fname.lower():
                    label = 0
                else:
                    continue
                self.img_paths.append(os.path.join(root, fname))
                self.labels.append(label)

        if len(self.img_paths) == 0:
            raise ValueError(f"No valid images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            random_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(random_idx)


class FolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        self.label_map = {'real': 1, 'spoof': 0}

        for label_name in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            label = self.label_map.get(label_name.lower())
            if label is None:
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.img_paths.append(os.path.join(label_dir, fname))
                    self.labels.append(label)

        if len(self.img_paths) == 0:
            raise ValueError(f"No valid images in {root_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())


class TextFileDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform

        with open(txt_file, 'r') as f:
            for line in f:
                path = line.strip()
                if not os.path.exists(path):
                    continue
                if 'real' in path.lower():
                    label = 1
                elif 'fake' in path.lower() or 'spoof' in path.lower():
                    label = 0
                else:
                    continue
                self.img_paths.append(path)
                self.labels.append(label)

        if len(self.img_paths) == 0:
            raise ValueError(f"No valid images found in {txt_file}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
