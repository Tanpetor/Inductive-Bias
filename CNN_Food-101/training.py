# training.py
import torch
from tqdm import tqdm
from metrics import calculate_metrics

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    return train_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            all_labels.append(labels)
            all_outputs.append(outputs)
    
    val_outputs = torch.cat(all_outputs)
    val_labels = torch.cat(all_labels)
    
    val_loss = val_loss / len(val_loader)
    val_acc, val_roc_auc = calculate_metrics(val_outputs, val_labels)
    
    return val_loss, val_acc, val_roc_auc