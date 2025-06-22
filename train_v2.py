import argparse
import torch
import torch.nn as nn
from dataset import TrainDataset, TestDataset
from model import MixStyleResCausalModel
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='experiment')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--lr', type=float, nargs=2, default=[0.001, 0.01])
    parser.add_argument('--input_shape', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--ops', type=bool, default=False)
    parser.add_argument('--norm', type=bool, default=False)
    return parser.parse_args()

def run_training(train_csv, test_csv, prefix, model_name, lr, input_shape, max_epoch, batch_size, pretrain, num_classes, ops, norm):

    train_dataset = TrainDataset(csv_file=train_csv, input_shape=input_shape)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = TestDataset(csv_file=test_csv, input_shape=input_shape)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = nn.DataParallel(MixStyleResCausalModel(
        model_name=model_name,
        pretrain=pretrain,
        num_classes=num_classes
    )).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr[0])
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(max_epoch):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epoch}")
        for batch in pbar:
            raw, labels, _ = batch  # Ignoramos el tercer elemento si no se usa
            raw = raw.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output, cf_output = model(raw, labels=labels, cf=ops, norm=norm)
                loss_1 = criterion(output, labels)
                loss_2 = criterion(output - cf_output, labels)
                loss = loss_1 + loss_2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=loss.item())

if __name__ == "__main__":
    args = parse_args()
    run_training(
        train_csv=args.training_csv,
        test_csv=args.test_csv,
        prefix=args.prefix,
        model_name=args.model_name,
        lr=args.lr,
        input_shape=args.input_shape,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        pretrain=args.pretrain,
        num_classes=args.num_classes,
        ops=args.ops,
        norm=args.norm
    )
