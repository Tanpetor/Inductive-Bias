import matplotlib.pyplot as plt

def plot_losses(all_results):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 2, 1)
    for res in all_results:
        plt.plot(res['train_loss'], '-', label=f"{res['name']} (train)")
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for res in all_results:
        plt.plot(res['val_loss'], '-', label=f"{res['name']} (val)")
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.savefig('full_metrics_comparison.png')
    plt.show()

def plot_metrics(all_results):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 2, 1)
    for res in all_results:
        plt.plot(res['val_accuracy'], '-', label=res['name'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for res in all_results:
        plt.plot(res['val_roc_auc'], '-', label=res['name'])
    plt.title('Validation ROC-AUC Score')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.savefig('full_metrics_comparison.png')
    plt.show()

def print_best_results(all_results):
    print("\n=== Best Configurations ===")
    print("Top by Validation Accuracy:")
    for res in sorted(all_results, key=lambda x: x['val_accuracy'][-1], reverse=True)[:3]:
        print(f"{res['name']}: {res['val_accuracy'][-1]:.4f}")

    print("\nTop by ROC-AUC:")
    for res in sorted(all_results, key=lambda x: x['val_roc_auc'][-1], reverse=True)[:3]:
        print(f"{res['name']}: {res['val_roc_auc'][-1]:.4f}")