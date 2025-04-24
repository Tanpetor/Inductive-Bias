import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_attention(model, image, device='mps'):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    img_tensor = image.unsqueeze(0).to(device)

    attention_maps = []

    def hook_fn(module, input, output):
        attention_maps.append(output.detach().cpu())

    hooks = []
    hooks.append(model.attention1.register_forward_hook(hook_fn))
    hooks.append(model.attention2.register_forward_hook(hook_fn))
    hooks.append(model.attention3.register_forward_hook(hook_fn))

    with torch.no_grad():
        output = model(img_tensor)

    for hook in hooks:
        hook.remove()

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axs[0].imshow(img_np)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    for i, attn_map in enumerate(attention_maps):
        attn = attn_map[0].mean(dim=0).numpy()
        axs[i + 1].imshow(attn, cmap='hot')
        axs[i + 1].set_title(f'Attention Layer {i + 1}')
        axs[i + 1].axis('off')

    plt.tight_layout()
    plt.show(block=False)


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    from sklearn.metrics import confusion_matrix
    import itertools

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show(block=False)
