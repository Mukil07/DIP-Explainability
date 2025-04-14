import pickle
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


with open("/scratch/mukilv2/cemformer/data_baseline.pkl", "rb") as f:
    data = pickle.load(f)

import pdb;pdb.set_trace()
# Print the loaded data
print(data)

vectors = torch.vstack(data['embeddings'][-17:])

# # Compute Pearson correlation matrix
# corr_matrix = np.corrcoef(vectors)

# # Convert to percentage
# corr_matrix *= 100  

# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm", cbar=True)
# plt.title("Pearson Correlation Matrix (%)")
# plt.savefig("vis_baseline.jpg")
# plt.show()

# plt.close()
import pdb;pdb.set_trace()
cos_sim_matrix = torch.nn.functional.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)

# Convert to percentage
cos_sim_matrix = cos_sim_matrix.numpy() * 100  

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix, annot=True, fmt=".1f", cmap="coolwarm", cbar=True)
plt.title("Cosine Similarity Matrix (%)")
plt.savefig("vis_cosine_baseline.jpg")
plt.show()

plt.close()

# cos_sim_matrix = torch.nn.functional.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=2)

# # Select a reference vector (e.g., first vector) similarity with all others
# ref_cos_sim = cos_sim_matrix[0].numpy()  # shape (17,)

# # Convert cosine similarity to percentages (range -100 to 100)
# ref_cos_sim_pct = ref_cos_sim * 100

# # Create a bar chart to visualize the cosine similarity percentages
# plt.figure(figsize=(10, 6))
# sns.barplot(x=np.arange(1, 18), y=ref_cos_sim_pct, palette="coolwarm")
# plt.xlabel("Vector Index")
# plt.ylabel("Cosine Similarity (%)")
# plt.title("Cosine Similarity of Vectors with Reference Vector (Vector 1)")
# plt.ylim(-100, 100)
# plt.axhline(0, color="black", linestyle="--")

# # Add similarity value labels above each bar
# for i, sim in enumerate(ref_cos_sim_pct):
#     # Adjust label placement based on whether the bar is positive or negative
#     offset = 3 if sim > 0 else -7
#     plt.text(i, sim + offset, f"{sim:.1f}%", ha="center", fontsize=10)
# plt.savefig("bar_baseline.jpg")
# plt.show()