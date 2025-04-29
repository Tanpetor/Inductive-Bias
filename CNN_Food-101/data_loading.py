from torchvision import datasets, transforms
from config import IMAGE_SIZE

train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_datasets():
    train_dataset = datasets.Food101(root='./data', split='train', transform=train_transform, download=True)
    val_dataset = datasets.Food101(root='./data', split='test', transform=val_transform, download=True)
    return train_dataset, val_dataset