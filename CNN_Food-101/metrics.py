import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from config import NUM_CLASSES

def calculate_metrics(outputs, labels):
    """Вычисляет accuracy и ROC-AUC"""
    _, preds = torch.max(outputs, 1)
    accuracy = (preds == labels).float().mean()
    
    labels_onehot = label_binarize(labels.cpu(), classes=range(NUM_CLASSES))
    outputs_softmax = F.softmax(outputs, dim=1).cpu().detach()
    
    try:
        roc_auc = roc_auc_score(labels_onehot, outputs_softmax, multi_class='ovr')
    except:
        roc_auc = 0.5 
    
    return accuracy.item(), roc_auc