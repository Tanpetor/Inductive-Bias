import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from celeba import CelebADataset


def prepare_dataloaders(dataset, batch_size=64, train_ratio=0.8, val_ratio=0.1):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


img_size = 64
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = CelebADataset(
    root_dir='/Users/tanpeter/Desktop/Deep Generative Learning/HW4/celeba',
    transform=transform
)
