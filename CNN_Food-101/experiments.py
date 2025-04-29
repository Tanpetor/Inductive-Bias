import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FoodClassifier
from data_loading import load_datasets
from training import train_epoch, validate
from config import device, EPOCHS, NUM_CLASSES

learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [64, 128]
weight_decays = [0.0001, 0.001]

all_results = []

def run_experiment(lr, batch_size, weight_decay):
    exp_name = f"lr={lr}_bs={batch_size}_wd={weight_decay}"
    print(f"\nStarting experiment: {exp_name}")
    
    train_dataset, val_dataset = load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    
    model = FoodClassifier(n_classes=NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5
    )
    
    results = {
        'name': exp_name,
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_roc_auc': [],
        'learning_rates': [] 
    }
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        results['train_loss'].append(train_loss)
        
        val_loss, val_acc, val_roc_auc = validate(model, val_loader, criterion, device)
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_acc)
        results['val_roc_auc'].append(val_roc_auc)
        
        current_lr = optimizer.param_groups[0]['lr']
        results['learning_rates'].append(current_lr)
        
        scheduler.step(val_loss)
        
        print(f"{exp_name} - Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"Val ROC-AUC: {val_roc_auc:.4f}, "
              f"LR: {current_lr:.2e}")
    
    all_results.append(results)
    return results

def run_all_experiments():
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for weight_decay in weight_decays:
                run_experiment(lr, batch_size, weight_decay)
    return all_results