import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
class plot_tsne():
    def __init__(self, perplexity=10, random_state=42, n_comp=2):
        self.tsne = TSNE(perplexity=perplexity, n_components=n_comp, random_state=random_state)

    def plot(self, features, labels, mode):

        features_np = torch.vstack(features).detach().cpu().numpy()


        #labels_list = [ten for unit in labels for ten in unit]
        labels_np = torch.vstack(labels).detach().cpu().numpy().squeeze()  # make sure it's 1D


        if mode == 'dipx':
            custom_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink', 'black']
        else:
            custom_colors = ['red', 'blue', 'green', 'purple', 'orange']

        cmap = ListedColormap(custom_colors)
        class_names = ['Straight', 'SlowDown', 'Left_Turn', 'Left_LC', 'Right_Turn', 'Right_LC', 'UTurn']
        colors = [cmap(i) for i in range(len(class_names))]

        reduced_features = self.tsne.fit_transform(features_np)

        fig, ax = plt.subplots(figsize=(8, 6))

        
        mask_non_black = labels_np != 8
        mask_black = labels_np == 8

        non_black_points = reduced_features[mask_non_black]
        non_black_labels = labels_np[mask_non_black]
        scatter = ax.scatter(
            non_black_points[:, 0],
            non_black_points[:, 1],
            c=non_black_labels,
            cmap=cmap,
            s=50,
        )

        # For the black points, use a list of unique markers.

        unique_markers = ['X', 's', '^', 'v', '<', '>', '1', '2', '3', '4',
                          'p', '*', 'h', 'H', 'D', 'd', 'P', '8','o' ]
        black_points = reduced_features[mask_black]

        n_black = black_points.shape[0]
        if n_black > len(unique_markers):
            raise ValueError(f"Need at least {n_black} unique markers for {n_black} black points. "
                             f"Currently, there are only {len(unique_markers)} markers.")

        #plot each black point individually with a different marker.
        for i, point in enumerate(black_points):
            ax.scatter(
                point[0],
                point[1],
                color='black',  #use black color for class 7.
                marker=unique_markers[i],
                s=80,
            )
        
        legend_handles = [
            mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=8, label=class_names[i])
            for i in range(len(class_names))
        ]

        anchor_legend = mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='Anchor Label')
        legend_handles.append(anchor_legend)

        ax.legend(handles=legend_handles, title="Classes", loc="best")

        colorbar = fig.colorbar(scatter, ax=ax, label='Classes')
        ax.set_title('t-SNE Visualization')

        return fig