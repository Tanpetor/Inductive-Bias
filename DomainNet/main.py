import torch
from datasets import get_train_loader, get_val_test_loaders
from CNN_model_IB import AttentionEnhancedModel
from CNN_model import BasicBlockNet
from train import train, evaluate
import torch.nn as nn
import torch.optim as optim


def main():
    train_domain = '/Users/tanpeter/Desktop/DomainNet/sketch'
    valtest_domain = '/Users/tanpeter/Desktop/DomainNet/quickdraw'
    img_size = 128
    batch_size = 64
    epochs = 40
    save_path = 'best_model.pth'
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, num_classes, class_names = get_train_loader(train_domain, img_size, batch_size)
    val_loader, test_loader = get_val_test_loaders(valtest_domain, img_size, batch_size, val_ratio=0.5)

    model = AttentionEnhancedModel(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    model = train(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path)

    test_acc = evaluate(model, test_loader, device)
    print(f"Best Model Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    main()
