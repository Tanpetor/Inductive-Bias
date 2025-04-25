from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_train_loader(train_dir, img_size=224, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    num_classes = len(dataset.classes)
    return loader, num_classes, dataset.classes


def get_val_test_loaders(valtest_dir, img_size=224, batch_size=64, val_ratio=0.5):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(valtest_dir, transform=transform)
    val_len = int(len(dataset) * val_ratio)
    test_len = len(dataset) - val_len
    val_set, test_set = random_split(dataset, [val_len, test_len], generator=None)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return val_loader, test_loader
