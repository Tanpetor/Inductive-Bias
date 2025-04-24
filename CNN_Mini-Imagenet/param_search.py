import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CNN_model import BasicBlockNet
from CNN_model_IB import AttentionEnhancedModel
from train import train_model, test_model
from dataset import MiniImageNetDataset, reorganize_dataset, ReorganizedDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import itertools


def param_search_adamw(model_type):
    lr_list = [1e-2, 1e-3, 1e-4]
    batch_size_list = [32, 64, 128]
    weight_decay_list = [1e-5, 1e-4]

    num_epochs = 100

    all_loss_curves = []
    all_acc_curves = []
    labels = []

    img_size = 64
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    mini_imagenet_path = 'autodl-tmp/archive/'
    train_dataset = MiniImageNetDataset('train.csv', mini_imagenet_path, transform)
    val_dataset = MiniImageNetDataset('val.csv', mini_imagenet_path, transform)
    test_dataset = MiniImageNetDataset('test.csv', mini_imagenet_path, transform)
    new_train_samples, new_val_samples, new_test_samples, all_classes = reorganize_dataset(
        train_dataset, val_dataset, test_dataset
    )
    new_train_dataset = ReorganizedDataset(new_train_samples, all_classes)
    new_val_dataset = ReorganizedDataset(new_val_samples, all_classes)
    new_test_dataset = ReorganizedDataset(new_test_samples, all_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0
    best_cfg = None
    results = []

    for lr in lr_list:
        for batch_size in batch_size_list:
            for wd in weight_decay_list:
                print(f"\n==== Training with lr={lr}, batch_size={batch_size}, weight_decay={wd} ====\n")
                train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                val_loader = DataLoader(new_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                test_loader = DataLoader(new_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                if model_type == "base":
                    model = BasicBlockNet()
                else:
                    model = AttentionEnhancedModel()
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

                model, train_losses, val_losses, train_accs, val_accs = train_model(
                    model, train_loader, val_loader,
                    criterion, optimizer, num_epochs=num_epochs, device=device,
                    par_search=f"lr:{lr}, bs:{batch_size}, wd{wd}"
                )
                all_loss_curves.append((train_losses, val_losses))
                all_acc_curves.append((train_accs, val_accs))
                labels.append(f"lr={lr}, bs={batch_size}, wd={wd}")
                acc = test_model(model, test_loader, criterion, all_classes, device=device)
                results.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'weight_decay': wd,
                    'test_acc': acc
                })

                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (lr, batch_size, wd)

    print("\n=== Summary of All Experiments ===")
    for r in results:
        print(r)
    print(
        f"\nBest Config: lr={best_cfg[0]}, batch_size={best_cfg[1]}, weight_decay={best_cfg[2]}, Best Test Accuracy: {best_acc:.2f}%")

    N = len(all_loss_curves)
    color_cycle = [plt.cm.hsv(i / N) for i in range(N)]

    plt.figure(figsize=(15, 8))
    for i, (train_losses, val_losses) in enumerate(all_loss_curves):
        color = color_cycle[i]
        plt.plot(train_losses, color=color, label=f'Train {labels[i]}', linestyle='-')
        plt.plot(val_losses, color=color, label=f'Val {labels[i]}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Different Hyperparameters')
    plt.legend(fontsize='small')
    plt.tight_layout()
    if model_type == "base":
        plt.savefig('loss_compare.png')
    else:
        plt.savefig('loss_compare_IB.png')
    plt.show(block=False)
    plt.close()

    plt.figure(figsize=(15, 8))
    color_cycle = [plt.cm.hsv(i / N) for i in range(len(all_acc_curves))]
    for i, (train_accs, val_accs) in enumerate(all_acc_curves):
        color = color_cycle[i]
        plt.plot(train_accs, color=color, label=f'Train {labels[i]}', linestyle='-')
        plt.plot(val_accs, color=color, label=f'Val {labels[i]}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves for Different Hyperparameters')
    plt.legend(fontsize='small')
    plt.tight_layout()
    if model_type == "base":
        plt.savefig('acc_compare.png')
    else:
        plt.savefig('acc_compare_IB.png')
    plt.show(block=False)


param_search_adamw("base")
param_search_adamw("IB")
