import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

class plot_tsne():

    def __init__(self,perplexity=10,random_state=42,n_comp=2):
        
        self.tsne = TSNE(perplexity=perplexity,n_components=n_comp, random_state=random_state)

    def plot(self,features,labels,mode):
        #import pdb;pdb.set_trace()
        features_np = torch.vstack(features).detach().cpu().numpy()
        labels = [ten for unit in labels for ten in unit]
        labels = torch.vstack(labels).detach().cpu().numpy()
        
        if mode == 'dipx':
            custom_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink','black']
        else:
            custom_colors = ['red', 'blue', 'green', 'purple', 'orange']
        tab5_custom = ListedColormap(custom_colors)
        reduced_features = self.tsne.fit_transform(features_np)
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap=tab5_custom, s=50)
        colorbar = fig.colorbar(scatter, ax=ax, label='Classes')
        ax.set_title('t-SNE Visualization')


        return fig
