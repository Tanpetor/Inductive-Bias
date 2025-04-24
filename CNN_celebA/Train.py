from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def get_15class_label(attributes):
    """
    4 - Bald
    5 - Bangs
    12 - Bushy_Eyebrows
    15 - Eyeglasses
    18 - Heavy_Makeup
    19 - High_Cheekbones
    20 - Male
    22 - Mustache
    24 - No_Beard (инвертируем для has_beard)
    29 - Rosy_Cheeks
    30 - Sideburns
    31 - Smiling
    32 - Straight_Hair
    33 - Wavy_Hair
    35 - Wearing_Hat
    39 - Young
    """
    # Основные атрибуты
    is_male = (attributes[:, 20] == 1)  # Male
    is_young = (attributes[:, 39] == 1)  # Young
    has_beard = (attributes[:, 24] == -1)  # No_Beard инвертирован
    is_smiling = (attributes[:, 31] == 1)  # Smiling
    has_eyeglasses = (attributes[:, 15] == 1)  # Eyeglasses
    has_hat = (attributes[:, 35] == 1)  # Wearing_Hat
    has_bangs = (attributes[:, 5] == 1)  # Bangs
    has_heavy_makeup = (attributes[:, 18] == 1)  # Heavy_Makeup
    has_wavy_hair = (attributes[:, 33] == 1)  # Wavy_Hair
    has_straight_hair = (attributes[:, 32] == 1)  # Straight_Hair
    has_rosy_cheeks = (attributes[:, 29] == 1)  # Rosy_Cheeks
    has_bushy_eyebrows = (attributes[:, 12] == 1)  # Bushy_Eyebrows
    is_bald = (attributes[:, 4] == 1)  # Bald
    has_mustache = (attributes[:, 22] == 1)  # Mustache
    has_sideburns = (attributes[:, 30] == 1)  # Sideburns

    labels = torch.zeros(attributes.shape[0], dtype=torch.long)

    # 0-2: Молодые с разными прическами
    labels[is_young & is_male & has_wavy_hair] = 0  # Young male wavy hair
    labels[is_young & is_male & has_straight_hair] = 1  # Young male straight hair
    labels[is_young & is_male & is_bald] = 2  # Young bald male

    # 3-5: Молодые женщины
    labels[is_young & ~is_male & has_bangs] = 3  # Young female with bangs
    labels[is_young & ~is_male & ~has_bangs & has_heavy_makeup] = 4  # Young female heavy makeup
    labels[is_young & ~is_male & ~has_bangs & ~has_heavy_makeup] = 5  # Young female natural

    # 6-8: Взрослые мужчины
    labels[~is_young & is_male & has_beard & has_mustache] = 6  # Adult bearded man with mustache
    labels[~is_young & is_male & has_beard & ~has_mustache] = 7  # Adult bearded man no mustache
    labels[~is_young & is_male & ~has_beard] = 8  # Adult clean-shaven man

    # 9-11: Взрослые женщины
    labels[~is_young & ~is_male & has_heavy_makeup & is_smiling] = 9  # Adult woman heavy makeup smiling
    labels[~is_young & ~is_male & has_heavy_makeup & ~is_smiling] = 10  # Adult woman heavy makeup not smiling
    labels[~is_young & ~is_male & ~has_heavy_makeup] = 11  # Adult woman natural

    # 12-14: Особые случаи
    labels[has_eyeglasses & is_male] = 12  # Male with glasses
    labels[has_eyeglasses & ~is_male] = 13  # Female with glasses
    labels[has_hat] = 14  # Any person with hat

    return labels


class_names = [
    "Young Male Wavy Hair",  # 0
    "Young Male Straight Hair",  # 1
    "Young Bald Male",  # 2
    "Young Female Bangs",  # 3
    "Young Female Heavy Makeup",  # 4
    "Young Female Natural",  # 5
    "Adult Bearded Man w/ Mustache",  # 6
    "Adult Bearded Man no Mustache",  # 7
    "Adult Clean-shaven Man",  # 8
    "Adult Woman Heavy Makeup Smiling",  # 9
    "Adult Woman Heavy Makeup Not Smiling",  # 10
    "Adult Woman Natural",  # 11
    "Male w/ Glasses",  # 12
    "Female w/ Glasses",  # 13
    "Person w/ Hat"  # 14
]


def train_15class_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    model = model.to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, targets in train_loop:
            images = images.to(device)
            labels = get_15class_label(targets['attributes']).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        val_loop = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, targets in val_loop:
                images = images.to(device)
                labels = get_15class_label(targets['attributes']).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, '/Users/tanpeter/Desktop/Deep Generative Learning/HW4/best_15class_model1.pth')

        print(f'\nEpoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show(block=False)

    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    return model
