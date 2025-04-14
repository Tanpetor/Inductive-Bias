import torch
import torch.nn as nn
import torch.optim as optim
from CNN_IB_model import InductiveBiasCelebA
from CNN_model import CelebA15ClassClassifier
from data_utils import prepare_dataloaders, dataset
from Train import train_15class_model

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(f'Using device: {device}')

#
model = InductiveBiasCelebA(n_classes=15)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train_loader, val_loader, test_loader = prepare_dataloaders(
    dataset,
    batch_size=128,
    train_ratio=0.8,
    val_ratio=0.1
)

model = train_15class_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device)
