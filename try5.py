import os
from torch.utils.data import Dataset
from PIL import Image

class CasiaFasdDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform

        for fname in os.listdir(img_dir):
            if not fname.endswith('.jpg'):
                continue
            label = 1 if 'real' in fname.lower() else 0
            self.img_paths.append(os.path.join(img_dir, fname))
            self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
