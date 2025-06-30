import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from model import SimpleModel
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


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

        print(f"Found {len(self.img_paths)} images in CASIA dataset {img_dir}")
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
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            random_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(random_idx)


class FolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        # Update to match your folder names
        self.label_map = {'real': 1, 'spoof': 0, 'fake': 0}  

        for label_name in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            label = self.label_map.get(label_name.lower())
            if label is None:
                print(f"Skipping unknown folder {label_name}")  # For debugging
                continue
            for fname in os.listdir(label_dir):
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): 
                    continue
                self.img_paths.append(os.path.join(label_dir, fname))
                self.labels.append(label)

        print(f"Found {len(self.img_paths)} images in folder dataset {root_dir}")
        if len(self.img_paths) == 0:
            raise ValueError(f"No valid images found in {root_dir}")


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
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            random_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(random_idx)

class TextFileDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform

        with open(txt_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            path = line.strip()
            if not os.path.isfile(path):
                print(f"Skipping missing file: {path}")
                continue
            if 'real' in path.lower():
                label = 1
            elif 'fake' in path.lower() or 'spoof' in path.lower():
                label = 0
            else:
                continue
            self.img_paths.append(path)
            self.labels.append(label)

        print(f"Found {len(self.img_paths)} images from {txt_file}")
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
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            random_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(random_idx)


def parse_args():
    parser = argparse.ArgumentParser(description='Anti-Spoofing Face Recognition Training')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to train folder or .txt file')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to val folder or .txt file')
    parser.add_argument('--prefix', type=str, default='experiment')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dataset_type', type=str, default='casia', choices=['casia', 'folder', 'text'])
    return parser.parse_args()


def plot_metrics(metrics, prefix):
    epochs = range(1, len(metrics['train_loss']) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Evolution")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics['val_acc'], 'g-', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics['lr'], 'm-', label='Learning Rate')
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()

    plt.tight_layout()
    os.makedirs("checkpoints", exist_ok=True)
    plt.savefig(f"checkpoints/{prefix}_metrics.png", dpi=300)
    plt.close()


def calculate_class_weights(labels, num_classes=2):
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * (class_counts + 1e-6))
    return torch.FloatTensor(class_weights)


def run_training(args):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if args.dataset_type == 'casia':
        train_dataset = CasiaFasdDataset(img_dir=args.train_dir, transform=train_transform)
        val_dataset = CasiaFasdDataset(img_dir=args.val_dir, transform=val_transform)
    elif args.dataset_type == 'folder':
        train_dataset = FolderDataset(root_dir=args.train_dir, transform=train_transform)
        val_dataset = FolderDataset(root_dir=args.val_dir, transform=val_transform)
    elif args.dataset_type == 'text':
        train_dataset = TextFileDataset(txt_file=args.train_dir, transform=train_transform)
        val_dataset = TextFileDataset(txt_file=args.val_dir, transform=val_transform)

    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)

    base_model = models.resnet18(weights=None)

    model = SimpleModel(
        num_classes=args.num_classes
    )
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.max_epoch * len(train_loader),
        pct_start=0.3, anneal_strategy='cos'
    )

    class_weights = calculate_class_weights(train_dataset.labels)
    criterion_main = nn.CrossEntropyLoss(weight=class_weights.cuda())

    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    metrics = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(args.max_epoch):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epoch}")
        running_loss = 0.0

        for images, labels in pbar:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion_main(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion_main(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        metrics['val_loss'].append(avg_val_loss)
        metrics['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = f"checkpoints/{args.prefix}_best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, best_path)
            print(f"Best model saved to {best_path} (Acc: {best_acc:.4f})")

        # Guardar checkpoint cada 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, f"checkpoints/{args.prefix}_epoch_{epoch+1}.pth")

    # Guardar modelo final
    final_path = f"checkpoints/{args.prefix}_final_model.pth"
    torch.save({
        'epoch': args.max_epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'args': vars(args)
    }, final_path)
    print(f"Training complete. Final model saved to {final_path}")

    plot_metrics(metrics, args.prefix)


if __name__ == "__main__":
    args = parse_args()
    print("Training configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    run_training(args)
