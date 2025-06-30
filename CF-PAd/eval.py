import argparse
import os
import hashlib
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from model import SimpleModel
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
            raise ValueError(f"No valid images in {img_dir}")
        print(f"[CASIA] {len(self.img_paths)} images loaded.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label, path
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

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
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                self.img_paths.append(os.path.join(label_dir, fname))
                self.labels.append(label)
        if len(self.img_paths) == 0:
            raise ValueError(f"No valid images in {root_dir}")
        print(f"[FOLDER] {len(self.img_paths)} images loaded.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label, path
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

class TextFileDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for path in lines:
            path = path.strip()
            if not os.path.isfile(path):
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
            raise ValueError(f"No valid images in {txt_file}")
        print(f"[TEXT] {len(self.img_paths)} images loaded.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label, path
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

def parse_args():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--dataset_type', choices=['casia', 'folder', 'text'], required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_csv', type=str, default=None)
    parser.add_argument('--save_misclassified', type=str, default=None)
    return parser.parse_args()

def sanitize_filename(s):
    return hashlib.md5(s.encode()).hexdigest()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if args.dataset_type == 'casia':
        dataset = CasiaFasdDataset(args.data, transform=transform)
    elif args.dataset_type == 'folder':
        dataset = FolderDataset(args.data, transform=transform)
    elif args.dataset_type == 'text':
        dataset = TextFileDataset(args.data, transform=transform)
    else:
        raise ValueError("Unsupported dataset type")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = SimpleModel(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    y_true, y_pred, paths = [], [], []

    with torch.no_grad():
        for images, labels, img_paths in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            paths.extend(img_paths)

            if args.save_misclassified:
                os.makedirs(args.save_misclassified, exist_ok=True)
                for i in range(len(labels)):
                    if preds[i] != labels[i]:
                        img = transforms.ToPILImage()(images[i].cpu())
                        fname = os.path.basename(img_paths[i])
                        img.save(os.path.join(args.save_misclassified, f"wrong_{fname}"))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["spoof", "real"]))
    print("Label distribution:", pd.Series(y_true).value_counts().sort_index().to_dict())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["spoof", "real"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    os.makedirs("confusion_matrices", exist_ok=True)
    unique_tag = sanitize_filename(f"{args.model_path}_{args.data}")
    cm_filename = os.path.join("confusion_matrices", f"confusion_{unique_tag}.png")
    plt.savefig(cm_filename)
    plt.close()
    print(f"Confusion matrix saved to {cm_filename}")

    os.makedirs("results", exist_ok=True)
    if args.save_csv:
        csv_path = args.save_csv
    else:
        csv_path = os.path.join("results", f"predictions_{unique_tag}.csv")

    pd.DataFrame({
        "path": paths,
        "true": y_true,
        "pred": y_pred
    }).to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        hter = (far + frr) / 2

        print(f"\nFalse Acceptance Rate (FAR): {far:.4f}")
        print(f"False Rejection Rate (FRR): {frr:.4f}")
        print(f"Half Total Error Rate (HTER): {hter:.4f}")

        os.makedirs("metrics", exist_ok=True)
        metrics_filename = os.path.join("metrics", f"metrics_{unique_tag}.csv")
        pd.DataFrame([{
            "model_path": args.model_path,
            "data_path": args.data,
            "FAR": far,
            "FRR": frr,
            "HTER": hter
        }]).to_csv(metrics_filename, index=False)
        print(f"Metrics saved to {metrics_filename}")
    else:
        print("Confusion matrix is not binary, cannot compute FAR/FRR/HTER.")

if __name__ == "__main__":
    main()
