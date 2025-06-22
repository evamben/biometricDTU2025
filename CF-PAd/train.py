# import argparse
# import os
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from torchvision.models import ResNet18_Weights
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# from model import ComplexModel
# from torch.cuda.amp import GradScaler, autocast
# from tqdm import tqdm
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
#     parser.add_argument('--train_dir', type=str, required=True)
#     parser.add_argument('--val_dir', type=str, required=True)
#     parser.add_argument('--prefix', type=str, default='experiment')
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--max_epoch', type=int, default=20)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--pretrained', action='store_true')
#     parser.add_argument('--num_classes', type=int, default=2)
#     parser.add_argument('--ops', type=str, default="", help="Counterfactual ops: dropout,shuffle,replace")
#     parser.add_argument('--ms_layers', type=str, default="layer1,layer2")
#     return parser.parse_args()


# def plot_metrics(train_losses, val_accuracies, prefix):
#     epochs = range(1, len(train_losses) + 1)
#     plt.figure(figsize=(10, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_losses, 'b', label='Training Loss')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training Loss")
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, val_accuracies, 'g', label='Validation Accuracy')
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Validation Accuracy")
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig(f"checkpoints/{prefix}_metrics.png")
#     plt.close()


# def run_training(train_dir, val_dir, prefix, lr, max_epoch, batch_size, pretrained, num_classes, ops, ms_layers):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
#     ])

#     train_dataset = CasiaFasdDataset(img_dir=train_dir, transform=transform)
#     val_dataset = CasiaFasdDataset(img_dir=val_dir, transform=transform)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     weights = ResNet18_Weights.DEFAULT if pretrained else None
#     base_model = models.resnet18(weights=weights)
#     model = ComplexModel(base_model=base_model, num_classes=num_classes, ms_layers=ms_layers.split(','))
#     model = nn.DataParallel(model).cuda()

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
#     # Calcular pesos de clase inversamente proporcionales a su frecuencia
#     labels = train_dataset.labels
#     num_samples = len(labels)
#     class_sample_counts = [labels.count(0), labels.count(1)]  # [Attack, Real]

#     # Evitar división por cero
#     epsilon = 1e-6
#     class_weights = [num_samples / (c + epsilon) for c in class_sample_counts]

#     # Convertir a tensor y normalizar
#     weight_tensor = torch.FloatTensor(class_weights).cuda()
#     criterion = nn.CrossEntropyLoss(weight=weight_tensor)

#     scaler = GradScaler()

#     best_acc = 0.0
#     os.makedirs('checkpoints', exist_ok=True)

#     cf_ops = [op.strip() for op in ops.split(',')] if ops else None

#     train_losses = []
#     val_accuracies = []

#     for epoch in range(max_epoch):
#         model.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epoch}")
#         total_loss = 0

#         for images, labels in pbar:
#             images, labels = images.cuda(), labels.cuda()
#             optimizer.zero_grad()

#             with autocast():
#                 out = model(images, labels=labels, cf_ops=cf_ops)

#                 if isinstance(out, tuple):
#                     output, cf_output = out
#                     loss1 = criterion(output, labels)
#                     loss2 = criterion(output - cf_output, labels)
#                     loss = loss1 + loss2
#                 else:
#                     loss = criterion(out, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             total_loss += loss.item()
#             pbar.set_postfix(loss=loss.item())

#         avg_loss = total_loss / len(train_loader)
#         train_losses.append(avg_loss)

#         # Validación
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.cuda(), labels.cuda()
#                 outputs = model(images, labels=labels, cf_ops=None)

#                 if isinstance(outputs, tuple):
#                     outputs = outputs[0]

#                 preds = torch.argmax(outputs, dim=1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)

#         acc = correct / total
#         val_accuracies.append(acc)
#         scheduler.step(acc)

#         print(f"🧪 Val Accuracy: {acc:.4f}")

#         # Guardar mejor modelo
#         if acc > best_acc:
#             best_acc = acc
#             best_path = f"checkpoints/{prefix}_best_model.pth"
#             torch.save(model.module.state_dict(), best_path)
#             print(f"📦 Modelo mejorado guardado en {best_path}")

#     # Guardar modelo final
#     final_path = f"checkpoints/{prefix}_final_model.pth"
#     torch.save(model.module.state_dict(), final_path)
#     print(f"✅ Entrenamiento completo. Modelo final guardado en {final_path}")

#     # Guardar métricas visuales
#     plot_metrics(train_losses, val_accuracies, prefix)


# if __name__ == "__main__":
#     args = parse_args()
#     run_training(
#         train_dir=args.train_dir,
#         val_dir=args.val_dir,
#         prefix=args.prefix,
#         lr=args.lr,
#         max_epoch=args.max_epoch,
#         batch_size=args.batch_size,
#         pretrained=args.pretrained,
#         num_classes=args.num_classes,
#         ops=args.ops,
#         ms_layers=args.ms_layers
#     )
# import argparse
# import os
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from torchvision.models import ResNet18_Weights
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# from model import ComplexModel
# from torch.cuda.amp import GradScaler, autocast
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import numpy as np
# from torch.nn import functional as F 
# class CasiaFasdDataset(Dataset):
#     def __init__(self, img_dir, transform=None):
#         self.img_paths = []
#         self.labels = []
#         self.transform = transform

#         for root, _, files in os.walk(img_dir):
#             for fname in files:
#                 if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     continue
                
#                 # Determinar si es real o fake basado en el nombre del archivo
#                 if 'real' in fname.lower():
#                     label = 1  # Real
#                 elif 'fake' in fname.lower() or 'spoof' in fname.lower():
#                     label = 0  # Fake
#                 else:
#                     continue  # Saltar archivos que no coincidan
                
#                 full_path = os.path.join(root, fname)
#                 self.img_paths.append(full_path)
#                 self.labels.append(label)

#         # Verificación
#         print(f"Found {len(self.img_paths)} images in {img_dir}")
#         if len(self.img_paths) == 0:
#             raise ValueError(f"No valid images found in {img_dir}")

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         try:
#             image = Image.open(img_path).convert("RGB")
#             label = self.labels[idx]

#             if self.transform:
#                 image = self.transform(image)

#             return image, label
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")
#             # Return a random valid image if there's an error
#             random_idx = torch.randint(0, len(self), (1,)).item()
#             return self.__getitem__(random_idx)

# def parse_args():
#     parser = argparse.ArgumentParser(description='Anti-Spoofing Face Recognition Training')
#     parser.add_argument('--train_dir', type=str, required=True, help='Directory with training images')
#     parser.add_argument('--val_dir', type=str, required=True, help='Directory with validation images')
#     parser.add_argument('--prefix', type=str, default='experiment', help='Experiment prefix for saving models')
#     parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
#     parser.add_argument('--max_epoch', type=int, default=50, help='Maximum number of epochs')
#     parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
#     parser.add_argument('--pretrained', action='store_true', help='Use pretrained ResNet weights')
#     parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
#     parser.add_argument('--ops', type=str, default="dropout,shuffle,noise", 
#                        help="Counterfactual ops: comma separated list of dropout,shuffle,noise")
#     parser.add_argument('--ms_layers', type=str, default="layer1,layer2,layer3",
#                        help="Layers to apply MixStyle: comma separated list")
#     parser.add_argument('--use_attention', action='store_true', help='Use channel attention modules')
#     parser.add_argument('--use_auxiliary', action='store_true', help='Use auxiliary branch for training')
#     parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
#     return parser.parse_args()

# def plot_metrics(metrics, prefix):
#     """Plot training and validation metrics"""
#     epochs = range(1, len(metrics['train_loss']) + 1)
#     plt.figure(figsize=(15, 5))
    
#     # Loss plot
#     plt.subplot(1, 3, 1)
#     plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
#     plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Loss Evolution")
#     plt.legend()

#     # Accuracy plot
#     plt.subplot(1, 3, 2)
#     plt.plot(epochs, metrics['val_acc'], 'g-', label='Validation Accuracy')
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Validation Accuracy")
#     plt.legend()

#     # Learning rate plot
#     plt.subplot(1, 3, 3)
#     plt.plot(epochs, metrics['lr'], 'm-', label='Learning Rate')
#     plt.xlabel("Epoch")
#     plt.ylabel("LR")
#     plt.title("Learning Rate Schedule")
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig(f"checkpoints/{prefix}_metrics.png", dpi=300)
#     plt.close()

# def calculate_class_weights(labels, num_classes=2):
#     """Calculate class weights for imbalanced datasets"""
#     class_counts = np.bincount(labels, minlength=num_classes)
#     total_samples = len(labels)
#     class_weights = total_samples / (num_classes * (class_counts + 1e-6))  # Smoothing
#     return torch.FloatTensor(class_weights)

# def run_training(args):
#     # Data augmentation and normalization
#     train_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Create datasets
#     train_dataset = CasiaFasdDataset(img_dir=args.train_dir, transform=train_transform)
#     val_dataset = CasiaFasdDataset(img_dir=args.val_dir, transform=val_transform)

#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
#                              shuffle=True, num_workers=8, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
#                            shuffle=False, num_workers=8, pin_memory=True)

#     # Initialize model
#     weights = ResNet18_Weights.DEFAULT if args.pretrained else None
#     base_model = models.resnet18(weights=weights)
#     model = ComplexModel(
#         base_model=base_model,
#         num_classes=args.num_classes,
#         ms_layers=args.ms_layers.split(','),
#         use_attention=args.use_attention,
#         use_auxiliary=args.use_auxiliary
#     )
#     model = nn.DataParallel(model).cuda()

#     # Optimizer and scheduler
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=args.lr,
#         total_steps=args.max_epoch * len(train_loader),
#         pct_start=0.3,
#         anneal_strategy='cos'
#     )

#     # Loss functions
#     class_weights = calculate_class_weights(train_dataset.labels)
#     criterion_main = nn.CrossEntropyLoss(weight=class_weights.cuda())
#     criterion_aux = nn.BCEWithLogitsLoss()
#     criterion_cf = nn.KLDivLoss(reduction='batchmean')

#     scaler = GradScaler()
#     best_acc = 0.0
#     os.makedirs('checkpoints', exist_ok=True)

#     cf_ops = [op.strip() for op in args.ops.split(',')] if args.ops else None

#     metrics = {
#         'train_loss': [],
#         'val_loss': [],
#         'val_acc': [],
#         'lr': []
#     }

#     for epoch in range(args.max_epoch):
#         model.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epoch}")
#         total_loss = 0

#         for images, labels in pbar:
#             images, labels = images.cuda(), labels.cuda()
#             optimizer.zero_grad()

#             with autocast():
#                 outputs = model(images, labels=labels, cf_ops=cf_ops)
                
#                 # Main classification loss
#                 loss_main = criterion_main(outputs['main'], labels)
#                 total_loss = loss_main
                
#                 # Auxiliary loss if enabled
#                 if args.use_auxiliary:
#                     aux_targets = labels.float().unsqueeze(1)
#                     loss_aux = criterion_aux(outputs['auxiliary'], aux_targets)
#                     total_loss += 0.5 * loss_aux
                
#                 # Counterfactual losses if enabled
#                 if cf_ops and 'cf' in outputs:
#                     for cf_type, cf_out in outputs['cf'].items():
#                         if cf_type == 'dropout':
#                             loss_cf = criterion_main(cf_out, labels)
#                         else:
#                             # For shuffle and noise, use KL divergence
#                             main_probs = F.softmax(outputs['main'], dim=1)
#                             cf_probs = F.softmax(cf_out, dim=1)
#                             loss_cf = criterion_cf(main_probs.log(), cf_probs)
#                         total_loss += 0.3 * loss_cf

#             scaler.scale(total_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()

#             pbar.set_postfix(loss=total_loss.item())

#         metrics['train_loss'].append(total_loss.item() / len(train_loader))
#         metrics['lr'].append(optimizer.param_groups[0]['lr'])

#         # Validation
#         model.eval()
#         val_loss = 0
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.cuda(), labels.cuda()
#                 outputs = model(images)
                
#                 loss = criterion_main(outputs['main'] if isinstance(outputs, dict) else outputs, labels)
#                 val_loss += loss.item()
                
#                 preds = torch.argmax(outputs['main'] if isinstance(outputs, dict) else outputs, dim=1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)

#         val_acc = correct / total
#         val_loss /= len(val_loader)
        
#         metrics['val_loss'].append(val_loss)
#         metrics['val_acc'].append(val_acc)

#         print(f"Epoch {epoch+1}/{args.max_epoch} | "
#               f"Train Loss: {metrics['train_loss'][-1]:.4f} | "
#               f"Val Loss: {val_loss:.4f} | "
#               f"Val Acc: {val_acc:.4f} | "
#               f"LR: {metrics['lr'][-1]:.2e}")

#         # Save best model
#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_path = f"checkpoints/{args.prefix}_best_model.pth"
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.module.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc,
#                 'args': vars(args)
#             }, best_path)
#             print(f"💾 Best model saved to {best_path} (Acc: {best_acc:.4f})")

#         # Save checkpoint every 5 epochs
#         if (epoch + 1) % 5 == 0:
#             checkpoint_path = f"checkpoints/{args.prefix}_epoch_{epoch+1}.pth"
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.module.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc,
#                 'args': vars(args)
#             }, checkpoint_path)

#     # Save final model
#     final_path = f"checkpoints/{args.prefix}_final_model.pth"
#     torch.save({
#         'epoch': args.max_epoch,
#         'model_state_dict': model.module.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'val_acc': val_acc,
#         'args': vars(args)
#     }, final_path)
#     print(f"🏁 Training complete. Final model saved to {final_path}")

#     # Plot metrics
#     plot_metrics(metrics, args.prefix)

# if __name__ == "__main__":
#     args = parse_args()
#     print("⚙️ Training configuration:")
#     for arg in vars(args):
#         print(f"{arg}: {getattr(args, arg)}")
    
#     run_training(args)
# import argparse
# import os
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from torchvision.models import ResNet18_Weights
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# from model import ComplexModel
# from torch.cuda.amp import GradScaler, autocast
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import numpy as np
# from torch.nn import functional as F


# class CasiaFasdDataset(Dataset):
#     def __init__(self, img_dir, transform=None):
#         self.img_paths = []
#         self.labels = []
#         self.transform = transform

#         for root, _, files in os.walk(img_dir):
#             for fname in files:
#                 if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     continue
#                 if 'real' in fname.lower():
#                     label = 1
#                 elif 'fake' in fname.lower() or 'spoof' in fname.lower():
#                     label = 0
#                 else:
#                     continue
#                 self.img_paths.append(os.path.join(root, fname))
#                 self.labels.append(label)

#         print(f"Found {len(self.img_paths)} images in CASIA dataset {img_dir}")
#         if len(self.img_paths) == 0:
#             raise ValueError(f"No valid images found in {img_dir}")

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         try:
#             image = Image.open(img_path).convert("RGB")
#             label = self.labels[idx]
#             if self.transform:
#                 image = self.transform(image)
#             return image, label
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")
#             random_idx = torch.randint(0, len(self), (1,)).item()
#             return self.__getitem__(random_idx)


# class FolderDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.img_paths = []
#         self.labels = []
#         self.transform = transform
#         self.label_map = {'real': 1, 'fake': 0}

#         for label_name in os.listdir(root_dir):
#             label_dir = os.path.join(root_dir, label_name)
#             if not os.path.isdir(label_dir):
#                 continue
#             label = self.label_map.get(label_name.lower())
#             if label is None:
#                 continue
#             for fname in os.listdir(label_dir):
#                 if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     continue
#                 self.img_paths.append(os.path.join(label_dir, fname))
#                 self.labels.append(label)

#         print(f"Found {len(self.img_paths)} images in folder dataset {root_dir}")
#         if len(self.img_paths) == 0:
#             raise ValueError(f"No valid images found in {root_dir}")

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         try:
#             image = Image.open(img_path).convert("RGB")
#             label = self.labels[idx]
#             if self.transform:
#                 image = self.transform(image)
#             return image, label
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")
#             random_idx = torch.randint(0, len(self), (1,)).item()
#             return self.__getitem__(random_idx)


# def parse_args():
#     parser = argparse.ArgumentParser(description='Anti-Spoofing Face Recognition Training')
#     parser.add_argument('--train_dir', type=str, required=True)
#     parser.add_argument('--val_dir', type=str, required=True)
#     parser.add_argument('--prefix', type=str, default='experiment')
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--max_epoch', type=int, default=5)
#     parser.add_argument('--batch_size', type=int, default=8)
#     parser.add_argument('--pretrained', action='store_true')
#     parser.add_argument('--num_classes', type=int, default=2)
#     parser.add_argument('--ops', type=str, default="dropout,shuffle,noise")
#     parser.add_argument('--ms_layers', type=str, default="layer1,layer2")
#     parser.add_argument('--use_attention', action='store_true')
#     parser.add_argument('--use_auxiliary', action='store_true')
#     parser.add_argument('--weight_decay', type=float, default=1e-4)
#     parser.add_argument('--dataset_type', type=str, default='casia', choices=['casia', 'folder'],
#                         help='Type of dataset: "casia" for filename-labels, "folder" for folder-structured')
#     return parser.parse_args()


# def plot_metrics(metrics, prefix):
#     epochs = range(1, len(metrics['train_loss']) + 1)
#     plt.figure(figsize=(15, 5))

#     plt.subplot(1, 3, 1)
#     plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
#     plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Loss Evolution")
#     plt.legend()

#     plt.subplot(1, 3, 2)
#     plt.plot(epochs, metrics['val_acc'], 'g-', label='Validation Accuracy')
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Validation Accuracy")
#     plt.legend()

#     plt.subplot(1, 3, 3)
#     plt.plot(epochs, metrics['lr'], 'm-', label='Learning Rate')
#     plt.xlabel("Epoch")
#     plt.ylabel("LR")
#     plt.title("Learning Rate Schedule")
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig(f"checkpoints/{prefix}_metrics.png", dpi=300)
#     plt.close()


# def calculate_class_weights(labels, num_classes=2):
#     class_counts = np.bincount(labels, minlength=num_classes)
#     total_samples = len(labels)
#     class_weights = total_samples / (num_classes * (class_counts + 1e-6))
#     return torch.FloatTensor(class_weights)


# def run_training(args):
#     train_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(0.2, 0.2, 0.2),
#         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     if args.dataset_type == 'casia':
#         train_dataset = CasiaFasdDataset(img_dir=args.train_dir, transform=train_transform)
#         val_dataset = CasiaFasdDataset(img_dir=args.val_dir, transform=val_transform)
#     elif args.dataset_type == 'folder':
#         train_dataset = FolderDataset(root_dir=args.train_dir, transform=train_transform)
#         val_dataset = FolderDataset(root_dir=args.val_dir, transform=val_transform)
#     else:
#         raise ValueError(f"Unknown dataset type: {args.dataset_type}")

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
#                               shuffle=True, num_workers=8, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
#                             shuffle=False, num_workers=8, pin_memory=True)

#     weights = ResNet18_Weights.DEFAULT if args.pretrained else None
#     base_model = models.resnet18(weights=weights)
#     model = ComplexModel(
#         base_model=base_model,
#         num_classes=args.num_classes,
#         ms_layers=args.ms_layers.split(','),
#         use_attention=args.use_attention,
#         use_auxiliary=args.use_auxiliary
#     )
#     model = nn.DataParallel(model).cuda()

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer, max_lr=args.lr, total_steps=args.max_epoch * len(train_loader),
#         pct_start=0.3, anneal_strategy='cos'
#     )

#     class_weights = calculate_class_weights(train_dataset.labels)
#     criterion_main = nn.CrossEntropyLoss(weight=class_weights.cuda())
#     criterion_aux = nn.BCEWithLogitsLoss()
#     criterion_cf = nn.KLDivLoss(reduction='batchmean')

#     scaler = GradScaler()
#     best_acc = 0.0
#     os.makedirs('checkpoints', exist_ok=True)

#     cf_ops = [op.strip() for op in args.ops.split(',')] if args.ops else None
#     metrics = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

#     for epoch in range(args.max_epoch):
#         model.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epoch}")
#         total_loss = 0

#         for images, labels in pbar:
#             images, labels = images.cuda(), labels.cuda()
#             optimizer.zero_grad()

#             with autocast():
#                 outputs = model(images, labels=labels, cf_ops=cf_ops)
#                 loss_main = criterion_main(outputs['main'], labels)
#                 total_loss = loss_main

#                 if args.use_auxiliary:
#                     aux_targets = labels.float().unsqueeze(1)
#                     loss_aux = criterion_aux(outputs['auxiliary'], aux_targets)
#                     total_loss += 0.5 * loss_aux

#                 if cf_ops and 'cf' in outputs:
#                     for cf_type, cf_out in outputs['cf'].items():
#                         if cf_type == 'dropout':
#                             loss_cf = criterion_main(cf_out, labels)
#                         else:
#                             main_probs = F.softmax(outputs['main'], dim=1)
#                             cf_probs = F.softmax(cf_out, dim=1)
#                             loss_cf = criterion_cf(main_probs.log(), cf_probs)
#                         total_loss += 0.3 * loss_cf

#             scaler.scale(total_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()
#             pbar.set_postfix(loss=total_loss.item())

#         metrics['train_loss'].append(total_loss.item() / len(train_loader))
#         metrics['lr'].append(optimizer.param_groups[0]['lr'])

#         # Validation
#         model.eval()
#         val_loss = 0
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.cuda(), labels.cuda()
#                 outputs = model(images)
#                 loss = criterion_main(outputs['main'] if isinstance(outputs, dict) else outputs, labels)
#                 val_loss += loss.item()

#                 preds = torch.argmax(outputs['main'] if isinstance(outputs, dict) else outputs, dim=1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)

#         val_acc = correct / total
#         val_loss /= len(val_loader)
#         metrics['val_loss'].append(val_loss)
#         metrics['val_acc'].append(val_acc)

#         print(f"Epoch {epoch+1} | Train Loss: {metrics['train_loss'][-1]:.4f} | "
#               f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_path = f"checkpoints/{args.prefix}_best_model.pth"
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.module.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc,
#                 'args': vars(args)
#             }, best_path)
#             print(f"💾 Best model saved to {best_path} (Acc: {best_acc:.4f})")

#         if (epoch + 1) % 5 == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.module.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc,
#                 'args': vars(args)
#             }, f"checkpoints/{args.prefix}_epoch_{epoch+1}.pth")

#     # Save final model
#     final_path = f"checkpoints/{args.prefix}_final_model.pth"
#     torch.save({
#         'epoch': args.max_epoch,
#         'model_state_dict': model.module.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'val_acc': val_acc,
#         'args': vars(args)
#     }, final_path)
#     print(f"🏁 Training complete. Final model saved to {final_path}")
#     plot_metrics(metrics, args.prefix)


# if __name__ == "__main__":
#     args = parse_args()
#     print("⚙️ Training configuration:")
#     for arg in vars(args):
#         print(f"{arg}: {getattr(args, arg)}")
#     run_training(args)


# # Dataset tipo carpeta (real/fake)
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
        self.label_map = {'real': 1, 'fake': 0}

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


def parse_args():
    parser = argparse.ArgumentParser(description='Anti-Spoofing Face Recognition Training')
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='experiment')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dataset_type', type=str, default='casia', choices=['casia', 'folder'],
                        help='Type of dataset: "casia" for filename-labels, "folder" for folder-structured')
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
            print(f"💾 Best model saved to {best_path} (Acc: {best_acc:.4f})")

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
    print(f"🏁 Training complete. Final model saved to {final_path}")

    plot_metrics(metrics, args.prefix)


if __name__ == "__main__":
    args = parse_args()
    print("⚙️ Training configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    run_training(args)
