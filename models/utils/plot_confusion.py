from sklearn.metrics import confusion_matrix
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

def confusion(all_labels, all_preds, mode, writer, epoch):

    if mode =='ego':
        cm = multilabel_confusion_matrix(all_labels, all_preds)
    else:
        cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap='Blues')

    fig.colorbar(cax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    ax.set_title(f'Confusion Matrix {mode}')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white')

    writer.add_figure(f'Confusion Matrix {mode}', fig,epoch)
    
