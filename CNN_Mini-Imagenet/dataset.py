import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import random


class MiniImageNetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        csv_path = os.path.join(root_dir, csv_file)
        self.data_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

        print(f"First 5 rows of CSV file {csv_file}:")
        print(self.data_frame.head())

        self.classes = sorted(self.data_frame.iloc[:, 1].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print(f"Number of classes: {len(self.classes)}")
        print(f"First few classes: {self.classes[:5]}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, 'images', self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.data_frame.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)

        return image, label


def reorganize_dataset(train_dataset, val_dataset, test_dataset, train_ratio=0.7, val_ratio=0.15):
    print("Starting dataset reorganization...")

    all_samples = []

    print("Collecting training samples...")
    for idx in tqdm(range(len(train_dataset))):
        image, label = train_dataset[idx]
        class_name = train_dataset.classes[label]
        all_samples.append((image, class_name))

    print("Collecting validation samples...")
    for idx in tqdm(range(len(val_dataset))):
        image, label = val_dataset[idx]
        class_name = val_dataset.classes[label]
        all_samples.append((image, class_name))

    print("Collecting test samples...")
    for idx in tqdm(range(len(test_dataset))):
        image, label = test_dataset[idx]
        class_name = test_dataset.classes[label]
        all_samples.append((image, class_name))

    print(f"Collected {len(all_samples)} samples in total")

    all_classes = sorted(list(set([class_name for _, class_name in all_samples])))
    print(f"Total number of detected classes: {len(all_classes)}")

    class_samples = {}
    for image, class_name in all_samples:
        if class_name not in class_samples:
            class_samples[class_name] = []
        class_samples[class_name].append(image)

    print(f"Total number of different classes: {len(class_samples)}")

    new_train, new_val, new_test = [], [], []

    for class_name, samples in class_samples.items():
        random.shuffle(samples)

        n = len(samples)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        if train_end == 0:
            train_end = 1
        if val_end <= train_end:
            val_end = train_end + 1
        if val_end >= n:
            val_end = n - 1

        new_train.extend([(img, class_name) for img in samples[:train_end]])
        new_val.extend([(img, class_name) for img in samples[train_end:val_end]])
        new_test.extend([(img, class_name) for img in samples[val_end:]])

    print(f"Dataset sizes after reorganization:")
    print(f"Training set: {len(new_train)} samples")
    print(f"Validation set: {len(new_val)} samples")
    print(f"Test set: {len(new_test)} samples")

    train_classes = set([cls for _, cls in new_train])
    val_classes = set([cls for _, cls in new_val])
    test_classes = set([cls for _, cls in new_test])

    print(f"Number of classes in training set: {len(train_classes)}")
    print(f"Number of classes in validation set: {len(val_classes)}")
    print(f"Number of classes in test set: {len(test_classes)}")

    return new_train, new_val, new_test, all_classes


class ReorganizedDataset(Dataset):
    def __init__(self, samples, all_classes):
        self.samples = samples
        self.classes = sorted(list(set([class_name for _, class_name in samples])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.all_classes = all_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, class_name = self.samples[idx]
        label = self.all_classes.index(class_name)
        return image, label
