import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from dataset import MiniImageNetDataset, reorganize_dataset, ReorganizedDataset
from CNN_model_IB import AttentionEnhancedModel
from CNN_model import BasicBlockNet
from train import train_model, test_model, show_examples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 64
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mini_imagenet_path = 'autodl-tmp/archive/'

    try:
        train_dataset = MiniImageNetDataset(
            csv_file='train.csv',
            root_dir=mini_imagenet_path,
            transform=transform
        )

        val_dataset = MiniImageNetDataset(
            csv_file='val.csv',
            root_dir=mini_imagenet_path,
            transform=transform
        )

        test_dataset = MiniImageNetDataset(
            csv_file='test.csv',
            root_dir=mini_imagenet_path,
            transform=transform
        )

        print("Datasets created successfully!")
    except Exception as e:
        print(f"Error creating datasets: {e}")

        for file in ['train.csv', 'val.csv', 'test.csv']:
            file_path = os.path.join(mini_imagenet_path, file)
            print(f"{file} exists: {os.path.exists(file_path)}")

        images_dir = os.path.join(mini_imagenet_path, 'images')
        print(f"images directory exists: {os.path.exists(images_dir)}")
        if os.path.exists(images_dir):
            print(f"Number of files in images directory: {len(os.listdir(images_dir))}")

    all_classes = []
    all_classes.extend(train_dataset.classes)
    all_classes.extend(val_dataset.classes)
    all_classes.extend(test_dataset.classes)
    all_classes = sorted(list(set(all_classes)))
    print(f"Total number of global classes: {len(all_classes)}")

    new_train_samples, new_val_samples, new_test_samples, all_classes = reorganize_dataset(
        train_dataset, val_dataset, test_dataset
    )

    new_train_dataset = ReorganizedDataset(new_train_samples, all_classes)
    new_val_dataset = ReorganizedDataset(new_val_samples, all_classes)
    new_test_dataset = ReorganizedDataset(new_test_samples, all_classes)

    new_train_loader = DataLoader(new_train_dataset, batch_size=64, shuffle=True, num_workers=0)
    new_val_loader = DataLoader(new_val_dataset, batch_size=64, shuffle=False, num_workers=0)
    new_test_loader = DataLoader(new_test_dataset, batch_size=64, shuffle=False, num_workers=0)

    new_train_classes = set([new_train_dataset.samples[i][1] for i in range(len(new_train_dataset))])
    new_val_classes = set([new_val_dataset.samples[i][1] for i in range(len(new_val_dataset))])
    new_test_classes = set([new_test_dataset.samples[i][1] for i in range(len(new_test_dataset))])

    model = AttentionEnhancedModel()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    model = train_model(model, new_train_loader, new_val_loader, criterion, optimizer, num_epochs=1, device=device)

    best_model = AttentionEnhancedModel()
    best_model = torch.load(
        'autodl-tmp/best_model_IB_1.pth',
        map_location=device, weights_only=False)
    test_model(best_model, new_test_loader, criterion, all_classes, device=device)

    show_examples(best_model, new_test_dataset, all_classes, n_examples=9, device=device)


if __name__ == "__main__":
    main()
