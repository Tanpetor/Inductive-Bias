import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='mps'):
    model = model.to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, labels in train_loop:
            images = images.to(device)
            labels = labels.to(device)

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
            for images, labels in val_loop:
                images = images.to(device)
                labels = labels.to(device)

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
            torch.save(model, 'best_model_IB_1.pth')

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
    plt.show()

    return model


def test_model(model, test_loader, criterion, all_classes, device='mps'):
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    test_loop = tqdm(test_loader, desc='Testing')
    with torch.no_grad():
        for images, labels in test_loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            test_loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    class_names = [str(cls) for cls in all_classes]
    present_classes = sorted(list(set([all_classes[i] for i in all_labels])))
    present_indices = [all_classes.index(cls) for cls in present_classes]
    present_class_names = [str(cls) for cls in present_classes]

    filtered_labels = []
    filtered_preds = []
    label_map = {idx: i for i, idx in enumerate(present_indices)}

    for label, pred in zip(all_labels, all_preds):
        if label in present_indices:
            filtered_labels.append(label_map[label])
            if pred in present_indices:
                filtered_preds.append(label_map[pred])
            else:
                filtered_preds.append(-1)

    unique_labels = sorted(list(set(filtered_labels)))
    if len(unique_labels) != len(present_class_names):
        used_classes = [present_indices[label] for label in unique_labels]
        present_class_names = [str(all_classes[idx]) for idx in used_classes]

    print(classification_report(filtered_labels, filtered_preds,
                                labels=range(len(present_class_names)),
                                target_names=present_class_names,
                                digits=3))

    return test_acc


def show_examples(model, dataset, all_classes, n_examples=9, device='mps'):
    model.eval()
    indices = np.random.choice(len(dataset), n_examples)

    plt.figure(figsize=(18, 15))
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            probs = torch.softmax(output, dim=1)
            top3_probs, top3_classes = torch.topk(probs, 3)

        image = (image - image.min()) / (image.max() - image.min())

        plt.subplot(3, 3, i + 1)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())

        class_names = [str(cls) for cls in all_classes]

        title = f'True: {class_names[label]}\nPred: {class_names[predicted.item()]}'
        for j in range(3):
            title += f"\n{class_names[top3_classes[0][j].item()]}: {top3_probs[0][j].item():.2f}"

        plt.title(title, fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
