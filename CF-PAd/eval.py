# import argparse
# import os
# import shutil
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# from model import ComplexModel
# from tqdm import tqdm

# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt


# class CasiaFasdDataset(Dataset):
#     def __init__(self, img_dir, transform=None):
#         self.img_paths = []
#         self.labels = []
#         self.transform = transform

#         for fname in os.listdir(img_dir):
#             if not fname.endswith('.jpg'):
#                 continue
#             label = 1 if 'real' in fname.lower() else 0
#             self.img_paths.append(os.path.join(img_dir, fname))
#             self.labels.append(label)

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str, required=True, help="Path to evaluation images")
#     parser.add_argument('--model_path', type=str, required=True, help="Path to the .pth model checkpoint")
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--num_classes', type=int, default=2)
#     parser.add_argument('--ms_layers', type=str, default="layer1,layer2", help="Comma-separated MixStyle layers")
#     parser.add_argument('--confusion_matrix_path', type=str, default="confusion_matrix.png", help="Path to save confusion matrix image")
#     return parser.parse_args()


# def plot_confusion_matrix(cm, classes, save_path):
#     plt.figure(figsize=(6,6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.title('Confusion Matrix')
#     plt.savefig(save_path)
#     plt.close()
#     print(f"📊 Confusion matrix saved to {save_path}")


# def evaluate(data_dir, model_path, batch_size, num_classes, ms_layers, confusion_matrix_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
#     ])

#     dataset = CasiaFasdDataset(img_dir=data_dir, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     base_model = models.resnet18(pretrained=False)
#     model = ComplexModel(base_model=base_model, num_classes=num_classes, ms_layers=ms_layers.split(','))

#     model = model.to(device)  # sin DataParallel para evitar problemas al cargar checkpoint

#     # Carga el checkpoint directamente en el modelo sin DataParallel
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)

#     model.eval()

#     all_labels = []
#     all_preds = []
#     all_probs = []

#     with torch.no_grad():
#         for images, labels in tqdm(dataloader, desc="Evaluating"):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images, labels=labels, cf_ops=None)
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]
#             probs = torch.softmax(outputs, dim=1)  # probabilidades para ROC/PR
#             preds = torch.argmax(probs, dim=1)

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_probs.extend(probs[:, 1].cpu().numpy())  # prob de clase 'Real'

#     acc = np.mean(np.array(all_preds) == np.array(all_labels))
#     print(f"✅ Evaluation Accuracy: {acc:.4f}")

#     target_names = ['Attack', 'Real']
#     report = classification_report(all_labels, all_preds, target_names=target_names)
#     print("\nClassification Report:\n", report)

#     # Confusion matrix
#     cm = confusion_matrix(all_labels, all_preds)
#     plot_confusion_matrix(cm, classes=target_names, save_path=confusion_matrix_path)

#     # ROC curve y AUC
#     fpr, tpr, _ = roc_curve(all_labels, all_probs)
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0,1])
#     plt.ylim([0,1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc='lower right')
#     plt.savefig('roc_curve.png')
#     plt.close()
#     print("📈 ROC curve saved as roc_curve.png")

#     # Precision-Recall curve
#     precision, recall, _ = precision_recall_curve(all_labels, all_probs)
#     plt.figure()
#     plt.plot(recall, precision, color='b', lw=2)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.savefig('precision_recall_curve.png')
#     plt.close()
#     print("📈 Precision-Recall curve saved as precision_recall_curve.png")

#     # Guardar imágenes mal clasificadas para inspección
#     misclassified_dir = 'misclassified_samples'
#     os.makedirs(misclassified_dir, exist_ok=True)
#     for idx, (img_path, true_label, pred_label) in enumerate(zip(dataset.img_paths, all_labels, all_preds)):
#         if true_label != pred_label:
#             dest_path = os.path.join(misclassified_dir, f"{idx}_true{true_label}_pred{pred_label}.jpg")
#             shutil.copy(img_path, dest_path)
#     print(f"🖼️ Saved misclassified images in {misclassified_dir}")


# if __name__ == "__main__":
#     args = parse_args()
#     evaluate(
#         data_dir=args.data_dir,
#         model_path=args.model_path,
#         batch_size=args.batch_size,
#         num_classes=args.num_classes,
#         ms_layers=args.ms_layers,
#         confusion_matrix_path=args.confusion_matrix_path
#     )
import argparse
import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from model import SimpleModel
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import seaborn as sns
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

        print(f"Found {len(self.img_paths)} images in {img_dir}")
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
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())


def parse_args():
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--data_dir', type=str, required=True, help="Path to evaluation images")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .pth model checkpoint")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset_type', type=str, default='casia', choices=['casia', 'folder'])
    parser.add_argument('--confusion_matrix_path', type=str, default="confusion_matrix.png")
    return parser.parse_args()


def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if args.dataset_type == 'casia':
        dataset = CasiaFasdDataset(img_dir=args.data_dir, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SimpleModel(num_classes=2)
    model = model.to(device)  # ❗️No usamos DataParallel

    # Carga del checkpoint sin DataParallel
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # ❗️Sin prefijo 'module.'

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # probabilidad de clase 'real'

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ Accuracy: {acc:.4f}")

    target_names = ['Fake', 'Real']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\nClassification Report:\n", report)

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, target_names, args.confusion_matrix_path)

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.close()
    print("📈 ROC curve saved as roc_curve.png")

    # PR Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
    plt.close()
    print("📈 Precision-Recall curve saved as precision_recall_curve.png")

    # Guardar imágenes mal clasificadas
    os.makedirs('misclassified_samples', exist_ok=True)
    for idx, (img_path, true_label, pred_label) in enumerate(zip(dataset.img_paths, all_labels, all_preds)):
        if true_label != pred_label:
            dest_path = os.path.join('misclassified_samples', f"{idx}_true{true_label}_pred{pred_label}.jpg")
            shutil.copy(img_path, dest_path)
    print("🖼️ Misclassified images saved in misclassified_samples/")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)