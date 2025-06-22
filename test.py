import argparse
import torch
import torch.nn as nn
from dataset import TestDataset
from model import MixStyleResCausalModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--feat_list', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=2)
    return parser.parse_args()

def run_test(test_csv, feat_list, model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = TestDataset(csv_file=test_csv, feat_file=feat_list)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = MixStyleResCausalModel(input_dim=512, num_classes=num_classes).to(device)

    # Cargar el modelo sin "module."
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for raw, labels, _ in tqdm(test_loader, desc="Testing"):
            raw = raw.to(device)
            labels = labels.to(device)

            outputs = model(raw)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = (all_preds == all_labels).mean()
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

    # Métricas completas para evitar "falsa confianza"
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print("🧮 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    args = parse_args()
    run_test(
        test_csv=args.test_csv,
        feat_list=args.feat_list,
        model_path=args.model_path,
        num_classes=args.num_classes
    )
