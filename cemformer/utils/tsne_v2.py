import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
class plot_tsne():
    def __init__(self, perplexity=10, random_state=42, n_comp=2):
        self.tsne = TSNE(perplexity=perplexity, n_components=n_comp, random_state=random_state)

    def plot(self, features, labels, mode):
        # Convert features and labels to numpy arrays.
        # (Assuming features is a list of tensors and labels is a nested list of tensors)
        features_np = torch.vstack(features).detach().cpu().numpy()
        # Flatten labels if needed (you already have nested lists/tensors)

        #labels_list = [ten for unit in labels for ten in unit]
        labels_np = torch.vstack(labels).detach().cpu().numpy().squeeze()  # make sure it's 1D

        # Define the colormap: for mode 'dipx' class 7 will be black.
        if mode == 'dipx':
            custom_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink', 'black']
        else:
            custom_colors = ['red', 'blue', 'green', 'purple', 'orange']

        cmap = ListedColormap(custom_colors)
        class_names = ['Straight', 'SlowDown', 'Left_Turn', 'Left_LC', 'Right_Turn', 'Right_LC', 'UTurn']
        colors = [cmap(i) for i in range(len(class_names))]
        # Apply t-SNE
        reduced_features = self.tsne.fit_transform(features_np)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create masks for non-black and black points.
        # Here we assume that the black class is represented by the integer 7.
        
        mask_non_black = labels_np != 7
        mask_black = labels_np == 7

        # Plot the non-black points all at once using scatter.
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
        # You can choose as many markers as needed. Here is an example list.
        unique_markers = ['X', 's', '^', 'v', '<', '>', '1', '2', '3', '4',
                          'p', '*', 'h', 'H', 'D', 'd', 'P','o' , '8']
        black_points = reduced_features[mask_black]

        # Check that you have enough unique markers
        n_black = black_points.shape[0]
        if n_black > len(unique_markers):
            raise ValueError(f"Need at least {n_black} unique markers for {n_black} black points. "
                             f"Currently, there are only {len(unique_markers)} markers.")

        # Plot each black point individually with a different marker.
        for i, point in enumerate(black_points):
            ax.scatter(
                point[0],
                point[1],
                color='black',  # Use black color for class 7.
                marker=unique_markers[i],
                s=80,
            )
        
        legend_handles = [
            mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None', markersize=8, label=class_names[i])
            for i in range(len(class_names))
        ]

        # 2. Legend for anchor (black) points
        anchor_legend = mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='Anchor Label')
        legend_handles.append(anchor_legend)

        # Add legend
        ax.legend(handles=legend_handles, title="Classes", loc="best")
        # Add a colorbar for the non-black points
        colorbar = fig.colorbar(scatter, ax=ax, label='Classes')
        ax.set_title('t-SNE Visualization')

        return fig